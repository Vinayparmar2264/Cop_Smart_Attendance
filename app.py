# app.py
import os
import errno
import json
import shutil
import re
import pickle
import base64
import time
import math
import requests
import io

from deepface import DeepFace  # Ensure this import is at the top of your file
from functools import wraps
from flask import request, jsonify, redirect, url_for
from flask_login import current_user, login_required
from flask import Flask, render_template, request, jsonify, Response, send_file, redirect, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user, UserMixin
from flask import redirect, url_for, abort, session, request, Response, stream_with_context,current_app


from sklearn.svm import SVC  # This import should already be at the top of your file

from datetime import datetime, timedelta, time as dt_time, date
from functools import wraps
import uuid
# Third-party Imports
import cv2
import dlib
import numpy as np
import flask
from flask import (Flask, render_template, Response, request, jsonify, session,
                   redirect,abort, url_for)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
# IMPORTANT: In a production environment, this secret key should be loaded from a
# secure location like an environment variable, not hardcoded.
app.secret_key = 'your_super_secret_corporate_key'

USERS_FILE = 'users.json'
# Root directory for storing all user-specific data, including teams and datasets.
USER_DATA_PATH = 'users_data'
# Ensure the root data directory exists upon application startup.
if not os.path.exists(USER_DATA_PATH):
    os.makedirs(USER_DATA_PATH)

try:
    # Dlib's pre-trained face detector (HOG-based). More accurate than Haar cascades.
    detector = dlib.get_frontal_face_detector()
    # Dlib's model for predicting the location of 68 facial landmarks.
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # Dlib's deep learning model for generating 128-dimensional face embeddings.
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    # OpenCV's Haar Cascade model. Faster but less accurate, used for real-time video feeds.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError(
            "Could not load Haar Cascade model. Please ensure the file 'haarcascade_frontalface_default.xml' is present.")
except Exception as e:
    print(f"FATAL ERROR: Could not load a required model file. {e}")
    print(
        "Please ensure 'shape_predictor_68_face_landmarks.dat', 'dlib_face_recognition_resnet_model_v1.dat', and 'haarcascade_frontalface_default.xml' are in the root directory.")
    # Exit the application if essential models are missing.
    exit()

# ---------------------------
# Helper utilities (preserved + additions)
# ---------------------------

def login_required(f):
    """Decorator function to ensure a user (leader) is logged in before accessing a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            # If the user is not logged in, redirect them to the login page.
            return redirect(url_for('auth', page='login'))
        # If the user is logged in, proceed with the original function.
        return f(*args, **kwargs)
    return decorated_function

def member_login_required(f):
    """Decorator function to ensure a team member is logged in before accessing their dashboard."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'member_info' not in session:
            # If member info is not in the session, redirect to the home page.
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function



@app.route('/member_marked_attendance')
@member_login_required
def member_marked_attendance():
    return render_template('member_marked_attendance.html', member=session.get('member_info'))



def get_user_path(username: str) -> str:
    """Constructs the absolute path to a user's (leader's) data directory."""
    return os.path.join(USER_DATA_PATH, username)

def get_team_path(username: str, team_name: str) -> str:
    """Constructs the absolute path to a specific team's directory under a leader."""
    return os.path.join(get_user_path(username), 'teams', team_name)

def get_team_data_path(username: str, team_name: str, data_type: str) -> str:
    """Constructs the path to a team's subdirectories (e.g., 'dataset', 'attendance_logs')."""
    return os.path.join(get_team_path(username, team_name), data_type)

def get_team_fields_config(username: str, team_name: str) -> list:
    """Returns team config fields, falling back to default if none found."""
    config_path = os.path.join(get_team_path(username, team_name), 'team_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    default_fields = [
        {"key": "enrollment_id", "label": "Enrollment ID", "editable": False, "permanent": True},
        {"key": "team_name", "label": "Team Name", "editable": False, "permanent": True},
        {"key": "enrollment_date", "label": "Join Date", "editable": False, "permanent": True},
        {"key": "leaving_date", "label": "Leave Date", "editable": False, "permanent": True},
        {"key": "position", "label": "Position / Role", "editable": True, "permanent": False},
        {"key": "email", "label": "Email Address", "editable": True, "permanent": False},
        {"key": "father_name", "label": "Father's Name", "editable": True, "permanent": False},
        {"key": "address", "label": "Full Address", "editable": True, "permanent": False},
    ]
    return default_fields

def norm_ymd(value):
    """Normalize a date-ish input to 'YYYY-MM-DD' string (safe guard)."""
    if value is None: return None
    if isinstance(value, (datetime, date)): return value.strftime('%Y-%m-%d')
    s = str(value)
    return s[:10]

# Ensure leader data tree exists when creating users via signup
def ensure_user_dirs(username):
    base = os.path.join(USER_DATA_PATH, username)
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
    teams_dir = os.path.join(base, 'teams')
    if not os.path.exists(teams_dir):
        os.makedirs(teams_dir, exist_ok=True)

# File path helpers for new features
def cycles_file(username):
    return os.path.join(get_user_path(username), 'leave_cycles.json')

def load_cycles(username):
    return json.load(open(cycles_file(username))) if os.path.exists(cycles_file(username)) else []

def save_cycles(username, cycles):
    with open(cycles_file(username), 'w') as f:
        json.dump(cycles, f, indent=4)

# A per-user cancellations file: stores team-specific cancellations (so calendar can show "for team X canceled")
def cancellations_file(username):
    return os.path.join(get_user_path(username), 'leave_cancellations.json')

def load_cancellations(username):
    return json.load(open(cancellations_file(username))) if os.path.exists(cancellations_file(username)) else []

def save_cancellations(username, data):
    with open(cancellations_file(username), 'w') as f:
        json.dump(data, f, indent=4)


#location based system paths

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return default if default is not None else {}

def save_json(path, data):
    ensure_folder(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------------
# Routes and core logic (preserved)
# ---------------------------


#location based attendance helpers


try:
    from flask_login import current_user
except Exception:
    current_user = None


def read_json_file(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def write_json_file(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def build_member_lookup(members_list):
    """Return dict enrollment_id -> member object, safe to use when computing reports."""
    return {m.get('enrollment_id'): m for m in members_list}



def haversine_meters(lat1, lon1, lat2, lon2):
    """Return distance between two lat/lon pairs in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def team_location_config_path(username, team_name):
    return os.path.join(get_team_path(username, team_name), "location_config.json")

def attendance_flags_path(username, team_name):
    return os.path.join(get_team_path(username, team_name), "attendance_flags")

def geocode_place_name(place_name):
    """Convert place name/pincode into lat/lon using Nominatim. Returns (lat, lon) or (None, None)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "AttendanceApp/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=6)
        if r.status_code == 200:
            data = r.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print("Geocoding failed:", e)
    return None, None

def read_attendance_flags(username, team_name, date_str):
    """Load per-day attendance flags (like location_mismatch)."""
    path = os.path.join(attendance_flags_path(username, team_name), f"{date_str}.json")
    return load_json(path, {})


# ------------------------------------------------------------
# ---------- LOCATION-BASED ATTENDANCE APIS ----------
# ------------------------------------------------------------
@app.route('/member/history')
@member_login_required
def member_history():
    """Renders the personal attendance history page for a logged-in member."""
    return render_template('member_history.html', member=session.get('member_info', {}))



@app.route("/api/save_team_location", methods=["POST"])
@login_required
def api_save_team_location():
    """Leader saves team location (from map click or text search)."""
    data = request.get_json(force=True)
    username = session['username']
    team = data.get("team_name")
    tolerance = int(data.get("tolerance_meters", 10))

    lat, lng, method, name = None, None, data.get("method"), data.get("name", "")

    if method == "map":
        lat, lng = float(data["lat"]), float(data["lng"])
    elif method == "text":
        lat, lng = geocode_place_name(name)
        if lat is None:
            return jsonify({"status": "error", "message": "Geocoding failed"}), 400
    else:
        return jsonify({"status": "error", "message": "Invalid method"}), 400

    cfg = {"lat": lat, "lng": lng, "tolerance_meters": tolerance, "name": name}
    save_json(team_location_config_path(username, team), cfg)
    return jsonify({"status": "success", "config": cfg})


@app.route("/api/get_team_location/<team>")
@login_required
def api_get_team_location(team):
    """Leader fetches saved team location."""
    username = session['username']
    cfg = load_json(team_location_config_path(username, team), {})
    return jsonify(cfg)


@app.route("/api/start_location_attendance", methods=["POST"])
@login_required
def api_start_location_attendance():
    """Leader enables location attendance flag for today."""
    data = request.get_json(force=True)
    username = session['username']
    team = data.get("team_name")
    date_str = datetime.now().strftime("%Y-%m-%d")
    ensure_folder(attendance_flags_path(username, team))
    path = os.path.join(attendance_flags_path(username, team), f"{date_str}.json")
    flags = load_json(path, {})
    flags["location_attendance_enabled"] = True
    save_json(path, flags)
    return jsonify({"status": "success", "message": f"Location attendance started for {team}"})


@app.route("/api/stop_location_attendance", methods=["POST"])
@login_required
def api_stop_location_attendance():
    """Leader disables location attendance flag for today."""
    data = request.get_json(force=True)
    username = session['username']
    team = data.get("team_name")
    date_str = datetime.now().strftime("%Y-%m-%d")
    ensure_folder(attendance_flags_path(username, team))
    path = os.path.join(attendance_flags_path(username, team), f"{date_str}.json")
    flags = load_json(path, {})
    flags["location_attendance_enabled"] = False
    save_json(path, flags)
    return jsonify({"status": "success", "message": f"Location attendance stopped for {team}"})





# @app.route("/api/get_todays_location_attendance/<team>")
# @login_required
# def api_get_todays_location_attendance(team):
#     """Leader views today's attendance scans from camera or location."""
#     # This route is used by both leader and member pages, so we need to get the leader's username correctly.
#     if 'username' in session:
#         username = session['username']  # For the logged-in leader
#     elif 'member_info' in session:
#         username = session['member_info']['leader_username']  # For the logged-in member
#     else:
#         return jsonify({"status": "error", "message": "Not authenticated"}), 401
#
#     date_str = datetime.now().strftime("%Y-%m-%d")
#
#     # CORRECTED PATH: Look in 'attendance_logs' for camera scans
#     path = os.path.join(get_team_data_path(username, team, 'attendance_logs'), f"{date_str}.json")
#
#     if not os.path.exists(path):
#         return jsonify([])  # Return empty list if no log file today
#
#     try:
#         with open(path, 'r') as f:
#             # Data is {"enroll_id_1": ["time1", "time2"], "enroll_id_2": ["time3"]}
#             all_scans = json.load(f)
#
#         # The frontend expects a list of objects, so we need to transform the data
#         transformed_rows = []
#         for enroll_id, times in all_scans.items():
#             for time in times:
#                 transformed_rows.append({
#                     "enrollment_id": enroll_id,
#                     "timestamp": f"{date_str} {time}",
#                     "accepted": True  # We'll just mark camera scans as accepted
#                 })
#         return jsonify(transformed_rows)
#     except Exception as e:
#         print(f"Error reading attendance logs: {e}")
#         return jsonify([])


#new one
@app.route("/api/get_todays_location_attendance/<team>")
@login_required
def api_get_todays_location_attendance(team):
    """
    Provides a live, calculated, final-report-style summary for all active
    members for the current day, based on the unified attendance logs.
    """
    username = session['username']
    today = datetime.now().date()

    members_file = os.path.join(get_team_path(username, team), 'team_members.json')
    all_members = read_json_file(members_file, {})

    todays_report = []
    for enroll_id, member_info in all_members.items():
        if member_info.get('status') == 'active':
            # Reuse the central calculation engine for perfect consistency
            member_daily_records = _calculate_final_attendance_for_member(username, team, enroll_id, today, today)
            if member_daily_records:
                report = member_daily_records[0]
                # Add distance info for this specific report if available
                report['distance_m'] = get_last_location_distance(username, team, enroll_id, today)
                todays_report.append(report)

    todays_report.sort(key=lambda x: x.get('name', ''))
    return jsonify(todays_report)


# Add this small helper function anywhere with your other helpers in app.py
def get_last_location_distance(username, team, enroll_id, date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    path = os.path.join(get_team_path(username, team), "location_attendance", f"{date_str}.json")
    rows = load_json(path, [])
    for row in reversed(rows):
        if row.get('enrollment_id') == enroll_id:
            return row.get('distance_m')
    return None
















@app.route('/api/member/match_location', methods=['POST'])
def api_member_match_location():
    """
    POST JSON:
      { "team_name": "team_x", "lat": 12.345678, "lng": 98.765432 }

    Response JSON:
      { status: "success", matched: true/false, distance_m: number, tolerance_m: number, message: "..." }
      OR
      { status: "error", message: "..." }
    """
    # AUTH: prefer member login decorator if you have one. If not, at least require session username.
    # If you have flask_login, use @login_required or current_user checks instead.
    username = session.get('username')
    if not username:
        return jsonify({'status':'error', 'message': 'Not authenticated'}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'status':'error', 'message': 'Expected JSON body'}), 400

    team_name = data.get('team_name') or data.get('team')  # allow either key
    lat = data.get('lat')
    lng = data.get('lng')

    if not team_name:
        return jsonify({'status':'error', 'message': 'team_name required'}), 400
    if lat is None or lng is None:
        return jsonify({'status':'error', 'message': 'lat and lng required'}), 400

    try:
        lat = float(lat)
        lng = float(lng)
    except Exception:
        return jsonify({'status':'error', 'message': 'lat/lng must be numeric'}), 400

    # load saved team location config (uses helper get_team_path and read_json_file)
    team_path = get_team_path(username, team_name)  # must exist in your codebase
    cfg_file = os.path.join(team_path, 'location_config.json')
    if not os.path.exists(cfg_file):
        return jsonify({'status':'error', 'message': 'No saved location configured for this team'}), 404

    cfg = read_json_file(cfg_file, {})  # helper used elsewhere in your code
    # Expecting cfg to contain { "lat": .., "lng": .., "tolerance_meters": 10, "name": "Office" }
    saved_lat = cfg.get('lat')
    saved_lng = cfg.get('lng')
    tol = cfg.get('tolerance_meters', cfg.get('tolerance', 10))

    if saved_lat is None or saved_lng is None:
        return jsonify({'status':'error', 'message': 'Saved team location is incomplete'}), 500

    try:
        saved_lat = float(saved_lat); saved_lng = float(saved_lng)
        tol = float(tol)
    except Exception:
        tol = 10.0

    dist = haversine_meters(lat, lng, saved_lat, saved_lng)

    matched = (dist <= tol)

    return jsonify({
        'status': 'success',
        'matched': bool(matched),
        'distance_m': round(dist, 2),
        'tolerance_m': tol,
        'team_location': {'lat': saved_lat, 'lng': saved_lng, 'name': cfg.get('name')},
        'message': 'Location matched' if matched else 'Location does NOT match'
    })



# --- ADD THIS CORRECTED ROUTE FOR THE MEMBER VIDEO FEED ---
# Make sure you have these imports at the top of your app.py file
from flask import Response, session

@app.route('/member/video_feed')
@member_login_required
def member_video_feed():
    """
    Provides the video stream for a logged-in member's attendance page.
    It retrieves the leader and team context from the member's session
    and uses the existing frame generator.
    """
    member_info = session.get('member_info', {})
    leader_username = member_info.get('leader_username')
    team_name = member_info.get('team_name')

    if not leader_username or not team_name:
        # This is a safeguard; it shouldn't happen if the member is properly logged in.
        return "Error: Could not determine team or leader context for video feed.", 400

    # This reuses your existing 'generate_frames' function, passing the correct
    # leader's username and team name from the member's session.
    return Response(generate_frames(leader_username, team_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')








# --- ADD THIS NEW ROUTE TO FIX THE "STOP ATTENDANCE" ERROR ---

@app.route('/api/member/mark_by_camera', methods=['POST'])
@member_login_required
def api_member_mark_by_camera():
    """
    Member-facing endpoint to finalize attendance after stopping the camera.
    This is called when the member clicks "Stop Attendance".
    """
    # Get member info securely from the session
    member_info = session.get('member_info', {})
    leader_username = member_info.get('leader_username')
    team_name = member_info.get('team_name')
    enrollment_id = member_info.get('enrollment_id')

    if not all([leader_username, team_name, enrollment_id]):
        return jsonify({"status": "error", "message": "Your session is invalid. Please log in again."}), 401

    # Reuse your existing robust log_attendance function
    result = log_attendance(leader_username, team_name, enrollment_id)
    cooldown_seconds = get_team_cooldown_seconds(leader_username, team_name)

    # Return a proper JSON response based on the result
    if result in ["LOGGED", "ALREADY_RECENT"]:
        return jsonify({
            "status": "success",
            "message": "Attendance captured successfully.",
            "result": result,
            "cooldown_seconds": cooldown_seconds
        })
    elif result == "COOLDOWN":
        return jsonify({
            "status": "cooldown",
            "message": f"Cooldown is active. Please wait.",
            "result": result,
            "cooldown_seconds": cooldown_seconds
        })
    elif result == "LEAVE_DAY":
        return jsonify({"status": "leave", "message": "Today is an announced leave day."})
    else: # Handles INACTIVE, NO_TEAM_FILE, etc.
        return jsonify({"status": "error", "message": "Could not log attendance.", "result": result}), 400


# --- ADD THESE TWO NEW HELPER FUNCTIONS ---

def get_member_today_scans(leader_username, team_name, enrollment_id):
    """A helper to get a member's raw scan times for today."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(get_team_data_path(leader_username, team_name, 'attendance_logs'), f"{today_str}.json")

    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, 'r') as f:
            all_logs = json.load(f)
        return all_logs.get(enrollment_id, [])
    except Exception:
        return []


def get_member_final_attendance_status(leader_username, team_name, enrollment_id):
    """A helper to calculate a member's final attendance status for today."""
    today = datetime.now().date()
    # We call your existing powerful analysis function for a single day.
    # Note: Ensure the 'compute_attendance_analysis' function exists in your app.py
    analysis = compute_attendance_analysis(leader_username, team_name, enrollment_id, today, today)

    if analysis.get("status") == "success" and analysis.get("days_analyzed") > 0:
        totals = analysis.get("totals", {})
        if totals.get("total_team_leaves", 0) > 0:
            return "On Leave"
        if totals.get("total_present", 0) > 0:
            return "Present"
        if totals.get("total_half_day", 0) > 0:
            return "Half Day"

    # Default to Absent if no other status is determined
    return "Absent"


# --- END OF NEW HELPER FUNCTIONS ---


# --- ADD THESE TWO NEW API ROUTES ---

@app.route('/api/member/get_today_scans', methods=['GET'])
@member_login_required
def api_member_get_today_scans():
    """API for a member to get their own scan times for today."""
    member_info = session.get('member_info', {})
    scans = get_member_today_scans(
        member_info.get('leader_username'),
        member_info.get('team_name'),
        member_info.get('enrollment_id')
    )
    return jsonify(scans)


@app.route('/api/member/get_final_attendance', methods=['GET'])
@member_login_required
def api_member_get_final_attendance():
    """API for a member to get their own final attendance status for today."""
    member_info = session.get('member_info', {})
    final_status = get_member_final_attendance_status(
        member_info.get('leader_username'),
        member_info.get('team_name'),
        member_info.get('enrollment_id')
    )
    # The frontend expects a table, so we'll simulate the structure it needs
    # by reusing the logic from your main final attendance route for a single day.
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = datetime.strptime(today_str, '%Y-%m-%d').date()

    # We call the main analysis function to get the full details
    result = compute_attendance_analysis(
        member_info.get('leader_username'),
        member_info.get('team_name'),
        member_info.get('enrollment_id'),
        start_date,
        start_date
    )

    if result.get('status') == 'success' and result.get('days_analyzed') > 0:
        # We extract the detailed daily record if it exists
        # Your 'compute_attendance_analysis' doesn't return day-by-day, so we build it
        # Let's call the detailed final attendance API instead, which is better
        return redirect(url_for('api_get_member_final_attendance',
                                team_name=member_info.get('team_name'),
                                enroll_id=member_info.get('enrollment_id'),
                                start=today_str,
                                end=today_str))

    return jsonify({"status": "error", "message": "Could not compute final attendance"}), 500


# --- END OF NEW API ROUTES ---




@app.route("/api/member_check_location", methods=["POST"])
@member_login_required
def api_member_check_location():
    """Member checks if their current location matches team's saved location."""
    data = request.get_json(force=True)
    member = session['member_info']
    username = member['leader_username']
    team = member['team_name']

    cfg = load_json(team_location_config_path(username, team), {})
    if not cfg:
        return jsonify({"status": "error", "message": "No team location set by leader."}), 400

    lat1, lng1 = float(data['lat']), float(data['lng'])
    lat2, lng2 = cfg.get("lat"), cfg.get("lng")
    tolerance = cfg.get("tolerance_meters", 10)

    dist = haversine_meters(lat1, lng1, lat2, lng2)
    matched = dist <= tolerance
    return jsonify({
        "status": "success",
        "distance": dist,
        "matched": matched,
        "tolerance": tolerance,
        "team_location": cfg
    })


# @app.route("/api/member_mark_attendance", methods=["POST"])
# @member_login_required
# def api_member_mark_attendance():
#     """Member marks attendance via location (in + out)."""
#     data = request.get_json(force=True)
#     member = session['member_info']
#     username = member['leader_username']
#     team = member['team_name']
#     enrollment_id = member['enrollment_id']
#     name = member['name']
#     now = datetime.now()
#     date_str = now.strftime("%Y-%m-%d")
#
#     # check location match
#     cfg = load_json(team_location_config_path(username, team), {})
#     if not cfg:
#         return jsonify({"status": "error", "message": "No team location set by leader."}), 400
#
#     lat1, lng1 = float(data['lat']), float(data['lng'])
#     lat2, lng2 = cfg.get("lat"), cfg.get("lng")
#     tolerance = cfg.get("tolerance_meters", 10)
#     dist = haversine_meters(lat1, lng1, lat2, lng2)
#     matched = dist <= tolerance
#
#     # Save attempt in team file
#     attempt = {
#         "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
#         "enrollment_id": enrollment_id,
#         "name": name,
#         "distance_m": dist,
#         "matched": matched,
#         "logged_status": "Present" if matched else "Absent"
#     }
#     path = os.path.join(get_team_path(username, team), "location_attendance", f"{date_str}.json")
#     rows = load_json(path, [])
#     rows.append(attempt)
#     save_json(path, rows)
#
#     # Save attendance flags (to be merged in final attendance)
#     flags = read_attendance_flags(username, team, date_str)
#     if not matched:
#         flags.setdefault("location_mismatch", []).append(enrollment_id)
#     save_json(os.path.join(attendance_flags_path(username, team), f"{date_str}.json"), flags)
#
#     return jsonify({"status": "success", "message": "Attendance logged", "attempt": attempt})

#new version
@app.route("/api/member_mark_attendance", methods=["POST"])
@member_login_required
def api_member_mark_attendance():
    """Member marks attendance via location. Now also logs to the main attendance log."""
    data = request.get_json(force=True)
    member = session['member_info']
    username = member['leader_username']
    team = member['team_name']
    enrollment_id = member['enrollment_id']
    name = member['name']
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # (Location matching logic remains the same)
    cfg = load_json(team_location_config_path(username, team), {})
    if not cfg:
        return jsonify({"status": "error", "message": "No team location set by leader."}), 400
    lat1, lng1 = float(data['lat']), float(data['lng'])
    lat2, lng2 = cfg.get("lat"), cfg.get("lng")
    tolerance = cfg.get("tolerance_meters", 10)
    dist = haversine_meters(lat1, lng1, lat2, lng2)
    matched = dist <= tolerance

    # --- NEW LOGIC ---
    # If the location is a match, call the central log_attendance function.
    # This unifies camera and location scans.
    if matched:
        log_attendance(username, team, enrollment_id)
    # --- END NEW LOGIC ---

    # Save the detailed location attempt for auditing purposes
    attempt = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "enrollment_id": enrollment_id,
        "name": name, "distance_m": dist, "matched": matched,
        "logged_status": "Logged" if matched else "Mismatch"
    }
    path = os.path.join(get_team_path(username, team), "location_attendance", f"{date_str}.json")
    rows = load_json(path, [])
    rows.append(attempt)
    save_json(path, rows)

    return jsonify({"status": "success", "message": "Attendance attempt recorded", "attempt": attempt})















# @app.route('/api/get_member_final_attendance/<team_name>/<enroll_id>')
# @member_login_required
# def api_get_member_final_attendance(team_name, enroll_id):
#     """
#     Return final attendance rows for a single member.
#     This version now correctly shows 'N/A' for future dates and dates
#     outside the member's employment period.
#     """
#     username = session['member_info']['leader_username']
#
#     start = request.args.get('start')
#     end = request.args.get('end') or start
#
#     try:
#         start_date = datetime.strptime(start, '%Y-%m-%d').date()
#         end_date = datetime.strptime(end, '%Y-%m-%d').date()
#     except (ValueError, TypeError):
#         return jsonify({'status': 'error', 'message': 'invalid date format; use YYYY-MM-DD'}), 400
#
#     # --- Load All Necessary Data Files Once ---
#     members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
#     members_data = read_json_file(members_file, {})
#     member_obj = members_data.get(enroll_id, {})
#     member_name = member_obj.get('name', '')
#
#     rules = read_json_file(os.path.join(get_team_path(username, team_name), 'attendance_rules.json'),
#                            get_default_attendance_rules())
#     team_leaves = read_json_file(os.path.join(get_team_path(username, team_name), 'leave_dates.json'), [])
#     member_leaves = read_json_file(os.path.join(get_team_path(username, team_name), 'member_leaves.json'), {})
#
#     try:
#         enrollment_date = datetime.strptime(member_obj.get('enrollment_date'), '%Y-%m-%d').date() if member_obj.get(
#             'enrollment_date') else None
#         leaving_date = datetime.strptime(member_obj.get('leaving_date'), '%Y-%m-%d').date() if member_obj.get(
#             'leaving_date') else None
#     except (ValueError, TypeError):
#         enrollment_date, leaving_date = None, None
#
#     records = []
#     today = datetime.now().date()
#     date_iter = start_date
#
#     while date_iter <= end_date:
#         date_str = date_iter.strftime('%Y-%m-%d')
#         report_item = {'date': date_str, 'enrollment_id': enroll_id, 'name': member_name, 'time_in': '-',
#                        'time_out': '-', 'working_hours': '0h 0m', 'late_coming': False, 'early_going': False,
#                        'status': 'Absent'}
#
#         # --- NEW LOGIC HIERARCHY ---
#         # Priority 1: Check if the date is in the future.
#         if date_iter > today:
#             report_item['status'] = 'N/A'
#             records.append(report_item)
#             date_iter += timedelta(days=1)
#             continue
#
#         # Priority 2: Check if the date is outside the employment period.
#         if (enrollment_date and date_iter < enrollment_date) or \
#                 (leaving_date and date_iter > leaving_date):
#             report_item['status'] = 'N/A'
#             records.append(report_item)
#             date_iter += timedelta(days=1)
#             continue
#
#         # Priority 3: Check for team-wide and personal leaves.
#         if date_str in team_leaves or (date_str in member_leaves and enroll_id in member_leaves.get(date_str, [])):
#             report_item['status'] = 'Leave'
#             records.append(report_item)
#             date_iter += timedelta(days=1)
#             continue
#
#         # Priority 4: Analyze attendance scans if it's a valid working day.
#         scans_file = os.path.join(get_team_data_path(username, team_name, 'attendance_logs'), f"{date_str}.json")
#         scans_by_member = read_json_file(scans_file, {})
#         scans_for_member = scans_by_member.get(enroll_id, [])
#
#         if len(scans_for_member) < 2:
#             report_item['status'] = 'Absent'
#             if len(scans_for_member) == 1:
#                 report_item['time_in'] = scans_for_member[0]
#                 report_item['notes'] = 'Single scan'
#         else:
#             # Full rules-based calculation from the leader's report
#             parsed = sorted([datetime.strptime(t, '%H:%M:%S') for t in scans_for_member])
#             time_in_dt, time_out_dt = parsed[0], parsed[-1]
#             total_seconds = (time_out_dt - time_in_dt).total_seconds()
#             hours, rem = divmod(total_seconds, 3600)
#             minutes, _ = divmod(rem, 60)
#
#             report_item.update({
#                 "time_in": time_in_dt.strftime('%H:%M:%S'),
#                 "time_out": time_out_dt.strftime('%H:%M:%S'),
#                 "working_hours": f"{int(hours)}h {int(minutes)}m",
#             })
#
#             # (Your complete and correct rules logic is applied here)
#             status, is_late, is_early = 'Absent', False, False
#             criteria_type = rules.get('criteria_type', 'time')
#             if criteria_type == 'hours':
#                 work_hours_float = total_seconds / 3600.0
#                 min_full = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_present', 8.0))
#                 min_early = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_early_go', 7.0))
#                 min_half = float(rules.get('half_day', {}).get('by_hours', {}).get('min_hours', 4.0))
#                 if work_hours_float >= min_full:
#                     status = 'Present'
#                 elif work_hours_float >= min_early:
#                     status = 'Present'; is_early = True
#                 elif work_hours_float >= min_half:
#                     status = 'Half Day'
#             else:  # Time-based rules
#                 def to_time(t_str):
#                     try:
#                         return dt_time.fromisoformat(t_str) if t_str else None
#                     except:
#                         return None
#
#                 safe_end = to_time(rules.get('half_day', {}).get('by_time', {}).get('in_time_safe_range_end'))
#                 hd_out = to_time(rules.get('half_day', {}).get('by_time', {}).get('required_out_time'))
#                 fd_early = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_early_go'))
#                 fd_full = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_present'))
#                 if safe_end and time_in_dt.time() > safe_end: is_late = True
#                 if fd_early and time_out_dt.time() < fd_early: is_early = True
#                 if fd_full and time_out_dt.time() >= fd_full:
#                     status = 'Present'
#                 elif fd_early and time_out_dt.time() >= fd_early:
#                     status = 'Present'
#                 elif hd_out and time_out_dt.time() >= hd_out:
#                     status = 'Half Day'
#
#             report_item['status'] = status
#             if status != 'Absent':
#                 report_item['late_coming'] = is_late
#                 report_item['early_going'] = is_early
#
#         records.append(report_item)
#         date_iter += timedelta(days=1)
#
#     return jsonify({'status': 'success', 'records': records})
# #end the location base system
# @app.route('/')
# def home():
#     """Redirects user to the appropriate dashboard or login page based on session."""
#     if 'username' in session:
#         return redirect(url_for('dashboard'))
#     if 'member_info' in session:
#         return redirect(url_for('member_dashboard'))
#     return redirect(url_for('auth', page='login'))





# new logic function for this
# REPLACE this function
@app.route('/api/get_member_final_attendance/<team_name>/<enroll_id>')
@member_login_required
def api_get_member_final_attendance(team_name, enroll_id):
    """ API for the member's personal history page. """
    username = session['member_info']['leader_username']
    start = request.args.get('start')
    end = request.args.get('end') or start
    try:
        start_date = datetime.strptime(start, '%Y-%m-%d').date()
        end_date = datetime.strptime(end, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'invalid date format'}), 400

    # Call the new central calculation engine
    records = _calculate_final_attendance_for_member(username, team_name, enroll_id, start_date, end_date)
    return jsonify({'status': 'success', 'records': records})















@app.route('/auth/<page>', methods=['GET', 'POST'])
def auth(page: str):
    if page == 'login':
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            users = json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}
            if username in users and 'password' in users[username] and check_password_hash(users[username]['password'],
                                                                                           password):
                session['username'] = username
                ensure_user_dirs(username)
                return redirect(url_for('dashboard'))
            else:
                return render_template('auth.html', page='login', error="Invalid username or password.")
        return render_template('auth.html', page='login')

    elif page == 'signup':
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            face_encoding_str = request.form.get('face_encoding')
            error = None
            if not re.match("^[A-Za-z0-9_]*$", username):
                error = "Username must be alphanumeric."
            elif len(password) < 8:
                error = "Password must be at least 8 characters long."
            elif password != confirm_password:
                error = "Passwords do not match."
            elif not face_encoding_str:
                error = "Face capture is required for sign up."
            else:
                users = json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}
                if username in users:
                    error = "Username already exists."
                else:
                    face_encoding = json.loads(face_encoding_str)
                    users[username] = {"password": generate_password_hash(password), "face_encoding": face_encoding}
                    with open(USERS_FILE, 'w') as f:
                        json.dump(users, f, indent=4)
                    os.makedirs(get_user_path(username), exist_ok=True)
                    flask.flash("Account created successfully! Please log in.", "success")
                    return redirect(url_for('auth', page='login'))
            return render_template('auth.html', page='signup', error=error)
        return render_template('auth.html', page='signup')

    elif page == 'forgot_password':
        error, username_validated, username_to_reset = None, False, None
        if request.method == 'POST':
            users = json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}
            if 'check_username' in request.form:
                username = request.form['username']
                if username in users:
                    username_validated = True
                    username_to_reset = username
                else:
                    error = "Username not found."
            elif 'reset_password' in request.form:
                username, new_password, confirm_password = request.form['username'], request.form['password'], \
                    request.form['confirm_password']
                if len(new_password) < 8:
                    error, username_validated, username_to_reset = "New password must be at least 8 characters.", True, username
                elif new_password != confirm_password:
                    error, username_validated, username_to_reset = "Passwords do not match.", True, username
                else:
                    users[username]['password'] = generate_password_hash(new_password)
                    with open(USERS_FILE, 'w') as f:
                        json.dump(users, f, indent=4)
                    flask.flash("Password reset successfully! Please log in.", "success")
                    return redirect(url_for('auth', page='login'))
        return render_template('auth.html', page='forgot_password', error=error, username_validated=username_validated,
                               username_to_reset=username_to_reset)
    return redirect(url_for('auth', page='login'))

@app.route('/member_login', methods=['GET', 'POST'])
def member_login():
    """Handles the initial step of member login by verifying text credentials."""
    if request.method == 'POST':
        leader_username, team_name, enrollment_id = request.form.get('leader_username'), request.form.get(
            'team_name'), request.form.get('enrollment_id')
        users = json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}
        if leader_username not in users:
            return render_template('auth.html', page='login', member_error="Leader username not found.")
        team_path = get_team_path(leader_username, team_name)
        if not os.path.exists(team_path):
            return render_template('auth.html', page='login', member_error="Team not found under this leader.")
        members_file = os.path.join(team_path, 'team_members.json')
        if not os.path.exists(members_file):
            return render_template('auth.html', page='login', member_error="No members found in this team.")
        with open(members_file, 'r') as f:
            members = json.load(f)
        if enrollment_id not in members:
            return render_template('auth.html', page='login', member_error="Enrollment ID not found in this team.")
        return render_template('auth.html', page='login', show_member_face_verify=True, leader_username=leader_username,
                               team_name=team_name, enrollment_id=enrollment_id)
    return redirect(url_for('auth', page='login'))

@app.route('/logout')
def logout():
    """Clears the session for either a leader or a member to log them out."""
    session.pop('username', None)
    session.pop('member_info', None)
    return redirect(url_for('auth', page='login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the main dashboard for the logged-in leader."""
    username = session['username']
    teams_path = os.path.join(get_user_path(username), 'teams')
    teams = [d for d in os.listdir(teams_path) if os.path.isdir(os.path.join(teams_path, d))] if os.path.exists(
        teams_path) else []
    return render_template('dashboard.html', username=username, teams=teams)

@app.route('/add_member')
@login_required
def add_member_page():
    """Renders the page for enrolling a new team member."""
    username = session['username']
    teams_path = os.path.join(get_user_path(username), 'teams')
    teams = [d for d in os.listdir(teams_path) if os.path.isdir(os.path.join(teams_path, d))] if os.path.exists(
        teams_path) else []
    return render_template('add_member.html', teams=teams)

@app.route('/mark_attendance')
@login_required
def mark_attendance():
    """Renders the page for live attendance marking."""
    username = session['username']
    teams_path = os.path.join(get_user_path(username), 'teams')
    teams = [d for d in os.listdir(teams_path) if os.path.isdir(os.path.join(teams_path, d))] if os.path.exists(
        teams_path) else []
    return render_template('mark_attendance.html', teams=teams)

@app.route('/member_dashboard')
@member_login_required
def member_dashboard():
    """Renders the dashboard for a logged-in team member."""
    return render_template('member_dashboard.html', member=session.get('member_info', {}))

@app.route('/leader/view_member_details/<path:team_name>/<enrollment_id>')
@login_required
def leader_view_member_details(team_name: str, enrollment_id: str):
    """Renders a detailed view of a specific member for the team leader."""
    username = session['username']
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if not os.path.exists(members_file):
        return "Member not found", 404
    with open(members_file, 'r') as f:
        members_data = json.load(f)
    member_info = members_data.get(enrollment_id)
    if not member_info:
        return "Member not found", 404
    member_info['team_name'] = team_name
    return render_template('leader_member_view.html', member=member_info)

@app.route('/video_feed_enroll')
@login_required
def video_feed_enroll():
    """Provides the video stream for the enrollment page."""
    return Response(generate_enroll_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_enroll_frames():
    """A generator function that captures frames from the camera for enrollment."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream for enrollment.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        guide_w, guide_h = 220, 280
        cv2.rectangle(frame, (cx - guide_w // 2, cy - guide_h // 2), (cx + guide_w // 2, cy + guide_h // 2),
                      (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


# --------- Add these helpers / replacements into app.py ----------


def _acquire_lock(lock_path, timeout=1.0, poll_interval=0.02):
    """Try to create a lock file atomically. Return file descriptor if created, else raise TimeoutError."""
    start = time.time()
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    mode = 0o600
    while True:
        try:
            fd = os.open(lock_path, flags, mode)
            return fd
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            if (time.time() - start) >= timeout:
                raise TimeoutError("Timeout acquiring lock")
            time.sleep(poll_interval)

def _release_lock(fd, lock_path):
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        os.remove(lock_path)
    except Exception:
        pass

def get_team_cooldown_seconds(username: str, team_name: str) -> int:
    team_path = get_team_path(username, team_name)
    cooldown_file = os.path.join(team_path, 'cooldown.json')
    default = 7200
    try:
        if os.path.exists(cooldown_file):
            data = json.load(open(cooldown_file, 'r'))
            s = int(data.get('seconds', default))
            if s < 0:
                return default
            return s
    except Exception as e:
        print(f"Warning: could not read cooldown for {team_name}: {e}")
    return default

# def log_attendance(username: str, team_name: str, enroll_id: str) -> str:
#     """
#     Robust logging:
#       - uses a tiny file-lock to avoid race writes
#       - avoids duplicate immediate writes (<1 second)
#       - respects team cooldown (get_team_cooldown_seconds)
#     Return values: "LOGGED", "COOLDOWN", "LEAVE_DAY", "NO_TEAM_FILE", "INACTIVE", "ALREADY_RECENT"
#     """
#     today_dt = datetime.now()
#     today_str = today_dt.strftime('%Y-%m-%d')
#
#     team_path = get_team_path(username, team_name)
#     members_file = os.path.join(team_path, 'team_members.json')
#     if not os.path.exists(members_file):
#         return "NO_TEAM_FILE"
#     try:
#         with open(members_file, 'r') as f:
#             all_members = json.load(f)
#     except Exception:
#         return "NO_TEAM_FILE"
#
#     member_info = all_members.get(enroll_id)
#     if not member_info or member_info.get('status') != 'active':
#         return "INACTIVE"
#
#     # Team-level leave check
#     leave_file = os.path.join(team_path, 'leave_dates.json')
#     try:
#         team_leaves = json.load(open(leave_file)) if os.path.exists(leave_file) else []
#     except Exception:
#         team_leaves = []
#     if today_str in team_leaves:
#         return "LEAVE_DAY"
#
#     # Prepare logs path and file
#     logs_path = get_team_data_path(username, team_name, 'attendance_logs')
#     os.makedirs(logs_path, exist_ok=True)
#     log_file = os.path.join(logs_path, f"{today_str}.json")
#     lock_file = log_file + '.lock'
#
#     # Acquire lock to perform read-modify-write safely
#     fd = None
#     try:
#         try:
#             fd = _acquire_lock(lock_file, timeout=1.0)
#         except TimeoutError:
#             # If cannot get lock quickly, fall back to reading file without lock but still safe-guard duplicates
#             pass
#
#         # Re-read file (fresh) after acquiring (or attempting) lock
#         try:
#             log_data = json.load(open(log_file)) if os.path.exists(log_file) else {}
#         except Exception:
#             log_data = {}
#
#         member_logs = log_data.get(enroll_id, [])
#
#         cooldown_seconds = get_team_cooldown_seconds(username, team_name)
#         current_time = datetime.now()
#
#         if not member_logs:
#             # First scan today -> record and return LOGGED
#             member_logs = [current_time.strftime('%H:%M:%S')]
#             log_data[enroll_id] = member_logs
#             try:
#                 with open(log_file, 'w') as f:
#                     json.dump(log_data, f, indent=4)
#             except Exception as e:
#                 print("Warning: failed to write log file:", e)
#             return "LOGGED"
#         else:
#             # Parse last recorded time
#             last_time_str = member_logs[-1]
#             try:
#                 last_time_obj = datetime.strptime(last_time_str, '%H:%M:%S').time()
#                 last_scan_dt = datetime.combine(current_time.date(), last_time_obj)
#                 # If last_scan_dt in future for some reason, adjust back a day
#                 if last_scan_dt > current_time:
#                     last_scan_dt = last_scan_dt - timedelta(days=1)
#             except Exception:
#                 # Parsing problem: be conservative and say COOLDOWN
#                 return "COOLDOWN"
#
#             seconds_since_last_scan = (current_time - last_scan_dt).total_seconds()
#
#             # If detection is a near-duplicate within 1 second -> ignore it (don't append)
#             if seconds_since_last_scan < 1.0:
#                 return "ALREADY_RECENT"
#
#             # Check configured cooldown
#             if seconds_since_last_scan < cooldown_seconds:
#                 return "COOLDOWN"
#
#             # Allowed to append a new scan (cooldown passed)
#             member_logs.append(current_time.strftime('%H:%M:%S'))
#             log_data[enroll_id] = member_logs
#             try:
#                 with open(log_file, 'w') as f:
#                     json.dump(log_data, f, indent=4)
#             except Exception as e:
#                 print("Warning: failed to write log file:", e)
#             return "LOGGED"
#     finally:
#         if fd is not None:
#             try:
#                 _release_lock(fd, lock_file)
#             except Exception:
#                 pass

# new one it is replaced due to solving intime and outime problem.

def log_attendance(username: str, team_name: str, enroll_id: str) -> str:
    """
    Robust logging with a hardcoded minimum duration between In and Out scans.
    """
    today_dt = datetime.now()
    today_str = today_dt.strftime('%Y-%m-%d')
    team_path = get_team_path(username, team_name)
    members_file = os.path.join(team_path, 'team_members.json')
    if not os.path.exists(members_file): return "NO_TEAM_FILE"

    all_members = read_json_file(members_file, {})
    if not all_members.get(enroll_id) or all_members[enroll_id].get('status') != 'active':
        return "INACTIVE"

    logs_path = get_team_data_path(username, team_name, 'attendance_logs')
    os.makedirs(logs_path, exist_ok=True)
    log_file = os.path.join(logs_path, f"{today_str}.json")
    lock_file = log_file + '.lock'
    fd = None
    try:
        fd = _acquire_lock(lock_file, timeout=1.0)
        log_data = read_json_file(log_file, {})
        member_logs = log_data.get(enroll_id, [])
        current_time = datetime.now()

        if len(member_logs) == 0:
            # First scan of the day -> This is IN-TIME
            member_logs.append(current_time.strftime('%H:%M:%S'))
            log_data[enroll_id] = member_logs
            write_json_file(log_file, log_data)
            return "LOGGED_IN"

        elif len(member_logs) == 1:
            # This is an attempt for an OUT-TIME scan
            last_time_str = member_logs[0]
            try:
                last_time_obj = datetime.strptime(last_time_str, '%H:%M:%S').time()
                last_scan_dt = datetime.combine(current_time.date(), last_time_obj)
            except:
                return "ERROR"

            seconds_since_last_scan = (current_time - last_scan_dt).total_seconds()

            # --- NEW FIX: Enforce a minimum duration ---
            # A hardcoded 10-minute (600 seconds) minimum between In and Out scans
            MINIMUM_DURATION_SECONDS = 600
            if seconds_since_last_scan < MINIMUM_DURATION_SECONDS:
                return "COOLDOWN"
            # --- END FIX ---

            # Also check the team's main cooldown setting
            cooldown_seconds = get_team_cooldown_seconds(username, team_name)
            if seconds_since_last_scan < cooldown_seconds:
                return "COOLDOWN"

            member_logs.append(current_time.strftime('%H:%M:%S'))
            log_data[enroll_id] = member_logs
            write_json_file(log_file, log_data)
            return "LOGGED_OUT"

        else:  # len(member_logs) >= 2
            return "ALREADY_COMPLETE"

    except TimeoutError:
        return "LOCKED"
    finally:
        if fd is not None:
            _release_lock(fd, lock_file)











# Manual endpoint: return cooldown_seconds in response so client can disable reliably
@app.route('/api/manual_mark_attendance', methods=['POST'])
def api_manual_mark_attendance():
    data = request.get_json() or {}
    team_name = data.get('team_name')
    enrollment_id = data.get('enrollment_id')

    if 'username' in session:
        username = session['username']
    elif 'member_info' in session:
        mi = session['member_info']
        username = mi.get('leader_username')
        if enrollment_id and enrollment_id != mi.get('enrollment_id'):
            return jsonify({"status": "error", "message": "Members may only mark their own attendance."}), 403
        if not enrollment_id:
            enrollment_id = mi.get('enrollment_id')
        if not team_name:
            team_name = mi.get('team_name')
    else:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if not team_name or not enrollment_id:
        return jsonify({"status": "error", "message": "Missing team or enrollment id."}), 400

    team_path = get_team_path(username, team_name)
    if not os.path.exists(team_path):
        return jsonify({"status": "error", "message": "Team not found."}), 404

    result = log_attendance(username, team_name, enrollment_id)
    cooldown_seconds = get_team_cooldown_seconds(username, team_name)
    if result == "LOGGED":
        return jsonify({"status": "success", "message": "Attendance logged.", "result": result, "cooldown_seconds": cooldown_seconds})
    elif result == "ALREADY_RECENT":
        # Treat as success for UI (first detection recorded already, no second write)
        return jsonify({"status": "success", "message": "Attendance already recorded just now.", "result": result, "cooldown_seconds": cooldown_seconds})
    elif result == "COOLDOWN":
        return jsonify({"status": "cooldown", "message": f"Cooldown active. Try after {cooldown_seconds} seconds.", "result": result, "cooldown_seconds": cooldown_seconds})
    elif result == "LEAVE_DAY":
        return jsonify({"status": "leave", "message": "Today is an announced leave day.", "result": result})
    elif result == "INACTIVE":
        return jsonify({"status": "error", "message": "Member inactive or not allowed to mark attendance.", "result": result})
    else:
        return jsonify({"status": "error", "message": "Unable to log attendance.", "result": result})


# GET cooldown (unchanged but ensure it returns stored seconds)
@app.route('/api/get_cooldown/<team_name>')
@login_required
def api_get_cooldown(team_name):
    username = session['username']
    team_path = get_team_path(username, team_name)
    if not os.path.exists(team_path):
        return jsonify({"status": "error", "message": "Team not found."}), 404
    seconds = get_team_cooldown_seconds(username, team_name)
    return jsonify({"status": "success", "seconds": seconds})


# SET cooldown: ensure this writes and returns the saved seconds
@app.route('/api/set_cooldown/<team_name>', methods=['POST'])
@login_required
def api_set_cooldown(team_name):
    username = session['username']
    body = request.get_json() or {}
    seconds = body.get('seconds')
    try:
        seconds = int(seconds)
        if seconds < 0 or seconds > 60 * 60 * 24 * 365:
            raise ValueError()
    except Exception:
        return jsonify({"status": "error", "message": "Invalid seconds value."}), 400
    team_path = get_team_path(username, team_name)
    if not os.path.exists(team_path):
        return jsonify({"status": "error", "message": "Team not found."}), 404
    cooldown_file = os.path.join(team_path, 'cooldown.json')
    try:
        with open(cooldown_file, 'w') as f:
            json.dump({"seconds": seconds}, f, indent=4)
        return jsonify({"status": "success", "message": f"Cooldown set to {seconds} seconds.", "seconds": seconds})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not write cooldown file: {e}"}), 500

# -----------------------------------------------------------------
# (End of app.py replacements)


@app.route('/video_feed/<path:team_name>')
@login_required
def video_feed(team_name: str):
    """Provides the video stream for the live attendance marking page."""
    return Response(generate_frames(session['username'], team_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- MODIFIED: generate_frames function to handle "COOLDOWN" status ---
# def generate_frames(username: str, team_name: str):
#     team_path = get_team_path(username, team_name)
#     try:
#         classifier = pickle.load(open(os.path.join(team_path, 'classifier.pkl'), 'rb'))
#         le = pickle.load(open(os.path.join(team_path, 'label_encoder.pkl'), 'rb'))
#         with open(os.path.join(team_path, 'team_members.json'), 'r') as f:
#             all_members = json.load(f)
#     except FileNotFoundError:
#         classifier, le, all_members = None, None, {}
#
#     leave_file = os.path.join(team_path, 'leave_dates.json')
#     is_leave_day = False
#     if os.path.exists(leave_file):
#         #modified area for cooldown
#         try:
#             with open(leave_file, 'r') as f:
#                 if datetime.now().strftime('%Y-%m-%d') in json.load(f):
#                     is_leave_day = True
#         except Exception:
#             is_leave_day = False
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success: break
#
#         if is_leave_day:
#             cv2.putText(frame, "TODAY IS A LEAVE DAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         elif classifier and le and all_members:
#             gray, rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 try:
#                     shape = sp(rgb, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
#                     encoding = np.array(facerec.compute_face_descriptor(rgb, shape)).reshape(1, -1)
#                     preds = classifier.predict_proba(encoding)[0]
#                     j = np.argmax(preds)
#                     confidence = preds[j]
#                 except Exception:
#                     continue
#                 display_name, color, status_text = "Unknown", (0, 0, 255), ""
#
#                 if confidence > 0.98:  # Your specified confidence from existing code
#                     enroll_id = le.inverse_transform([j])[0]
#                     display_name = all_members.get(enroll_id, {}).get('name', 'Unknown')
#                     log_status = log_attendance(username, team_name, enroll_id)
#                     if log_status == "LOGGED":
#                         color = (0, 255, 0)
#                         status_text = " (Logged)"
#                     elif log_status == "INACTIVE":
#                         color = (0, 165, 255)
#                         status_text = " (Inactive)"
#                     elif log_status == "COOLDOWN":  # NEW STATUS
#                         color = (255, 191, 0)  # Light blue color
#                         status_text = " (Cooldown)"
#                     else:
#                         color = (0, 255, 255)  # Yellow for other states like LEAVE_DAY
#
#                 text = f"{display_name}{status_text}"
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret: continue
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#     cap.release()




#new generate frame function
# def generate_frames(username: str, team_name: str):
#     team_path = get_team_path(username, team_name)
#     try:
#         classifier = pickle.load(open(os.path.join(team_path, 'classifier.pkl'), 'rb'))
#         le = pickle.load(open(os.path.join(team_path, 'label_encoder.pkl'), 'rb'))
#         all_members = read_json_file(os.path.join(team_path, 'team_members.json'), {})
#     except FileNotFoundError:
#         classifier, le, all_members = None, None, {}
#
#     cap = cv2.VideoCapture(0)
#     tracker, tracking_info, frame_count = None, {}, 0
#     DETECTION_INTERVAL = 10
#
#     while True:
#         success, frame = cap.read()
#         if not success: break
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         if tracker is None:
#             if frame_count % DETECTION_INTERVAL == 0 and classifier is not None:
#                 boxes = detector(rgb, 0)
#                 if len(boxes) > 0:
#                     box = boxes[0]
#                     face_img = rgb[box.top():box.bottom(), box.left():box.right()]
#                     try:
#                         embedding_obj = DeepFace.represent(img_path=face_img, model_name="ArcFace",
#                                                            enforce_detection=False)
#                         embedding = embedding_obj[0]["embedding"]
#                         preds = classifier.predict_proba([embedding])[0]
#                         j = np.argmax(preds)
#                         confidence = preds[j]
#                         if confidence > 0.85:
#                             enroll_id = le.inverse_transform([j])[0]
#                             tracking_info = {'id': enroll_id,
#                                              'name': all_members.get(enroll_id, {}).get('name', 'Unknown')}
#                             tracker = dlib.correlation_tracker()
#                             tracker.start_track(rgb, box)
#                         else:
#                             tracking_info = {'name': 'Unknown'}
#                     except Exception:
#                         tracking_info = {'name': 'Unknown'}
#         else:
#             confidence = tracker.update(rgb)
#             if confidence >= 8.5:
#                 pos = tracker.get_position()
#                 (x, y, r, b) = (int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
#                 display_name = tracking_info.get('name', 'Unknown')
#                 color, status_text = (0, 0, 255), ""
#                 if display_name != 'Unknown':
#                     log_status = log_attendance(username, team_name, tracking_info['id'])
#                     if log_status == "LOGGED":
#                         color, status_text = (0, 255, 0), " (Logged)"
#                     elif log_status == "COOLDOWN":
#                         color, status_text = (255, 191, 0), " (Cooldown)"
#                     else:
#                         color, status_text = (0, 255, 255), " (Already Scanned)"
#                 text = f"{display_name}{status_text}"
#                 cv2.rectangle(frame, (x, y), (r, b), color, 2)
#                 cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#             else:
#                 tracker = None
#
#         frame_count += 1
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret: continue
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#     cap.release()


#new due to intime outtime problem
def generate_frames(username: str, team_name: str):
    team_path = get_team_path(username, team_name)
    try:
        classifier = pickle.load(open(os.path.join(team_path, 'classifier.pkl'), 'rb'))
        le = pickle.load(open(os.path.join(team_path, 'label_encoder.pkl'), 'rb'))
        all_members = read_json_file(os.path.join(team_path, 'team_members.json'), {})
    except FileNotFoundError:
        classifier, le, all_members = None, None, {}

    cap = cv2.VideoCapture(0)
    tracker, tracking_info, frame_count = None, {}, 0
    DETECTION_INTERVAL = 10

    while True:
        success, frame = cap.read()
        if not success: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if tracker is None:
            if frame_count % DETECTION_INTERVAL == 0 and classifier is not None:
                boxes = detector(rgb, 0)
                if len(boxes) > 0:
                    box = boxes[0]
                    face_img = rgb[box.top():box.bottom(), box.left():box.right()]
                    try:
                        embedding_obj = DeepFace.represent(img_path=face_img, model_name="ArcFace",
                                                           enforce_detection=False)
                        embedding = embedding_obj[0]["embedding"]
                        preds = classifier.predict_proba([embedding])[0]
                        j = np.argmax(preds)
                        confidence = preds[j]
                        if confidence > 0.6:
                            enroll_id = le.inverse_transform([j])[0]
                            tracking_info = {'id': enroll_id,
                                             'name': all_members.get(enroll_id, {}).get('name', 'Unknown')}
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(rgb, box)
                        else:
                            tracking_info = {'name': 'Unknown'}
                    except Exception:
                        tracking_info = {'name': 'Unknown'}
        else:
            confidence = tracker.update(rgb)
            if confidence >= 8.5:
                pos = tracker.get_position()
                (x, y, r, b) = (int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))

                display_name = tracking_info.get('name', 'Unknown')
                color, status_text = (0, 0, 255), ""

                if display_name != 'Unknown':
                    log_status = log_attendance(username, team_name, tracking_info['id'])

                    if log_status == "LOGGED_IN":
                        color, status_text = (0, 255, 0), " (In-Time Logged)"
                    elif log_status == "LOGGED_OUT":
                        color, status_text = (0, 255, 0), " (Out-Time Logged)"
                    elif log_status == "COOLDOWN":
                        color, status_text = (255, 191, 0), " (Cooldown)"
                    elif log_status == "ALREADY_COMPLETE":
                        color, status_text = (0, 255, 255), " (Day Complete)"
                    # For other statuses like INACTIVE, NO_TEAM_FILE, etc., we can show a generic message or no message.
                    # This covers the main feedback loop.

                text = f"{display_name}{status_text}"
                cv2.rectangle(frame, (x, y), (r, b), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                tracker = None

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/api/create_team', methods=['POST'])
@login_required
def api_create_team():
    """API endpoint for a leader to create a new team."""
    username = session['username']
    team_name = request.json.get('team_name')
    if not team_name or not re.match("^[A-Za-z0-9_ ]*$", team_name):
        return jsonify({"status": "error", "message": "Invalid team name."})
    team_path = get_team_path(username, team_name)
    if os.path.exists(team_path):
        return jsonify({"status": "error", "message": "Team name already exists."})
    os.makedirs(get_team_data_path(username, team_name, 'dataset'), exist_ok=True)
    os.makedirs(get_team_data_path(username, team_name, 'attendance_logs'), exist_ok=True)
    with open(os.path.join(team_path, 'team_info.json'), 'w') as f:
        json.dump({'created_date': datetime.now().strftime('%Y-%m-%d')}, f)
    # Initialize default team files
    with open(os.path.join(team_path, 'team_members.json'), 'w') as f:
        json.dump({}, f, indent=4)
    with open(os.path.join(team_path, 'leave_dates.json'), 'w') as f:
        json.dump([], f, indent=4)
    with open(os.path.join(team_path, 'team_rules.json'), 'w') as f:
        json.dump({}, f, indent=4)
    return jsonify({"status": "success", "message": f"Team '{team_name}' created."})

@app.route('/api/get_team_details/<path:team_name>')
@login_required
def api_get_team_details(team_name):
    """API endpoint to fetch the list of members for a given team."""
    username = session['username']
    team_path = get_team_path(username, team_name)
    members_file = os.path.join(team_path, 'team_members.json')
    members = []
    if os.path.exists(members_file):
        with open(members_file, 'r') as f:
            members_data = json.load(f)
            members = [member for member in members_data.values() if member.get('status') != 'deleted']
    return jsonify({'members': members})



# --- Member-level leave helpers (place near other helpers) ---
def get_member_leaves_path(username: str, team_name: str) -> str:
    """Return path to member_leaves.json for a team."""
    return os.path.join(get_team_path(username, team_name), 'member_leaves.json')

def load_member_leaves(username: str, team_name: str) -> dict:
    """Load mapping date_str -> list of enrollment_ids who have approved leave on that date."""
    path = get_member_leaves_path(username, team_name)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # normalize to lists
                return {k: list(v) for k, v in data.items()}
        except Exception as e:
            print(f"Warning: failed to load member_leaves.json: {e}")
            return {}
    return {}

def save_member_leaves(username: str, team_name: str, data: dict):
    """Persist member-level leaves (atomic write)."""
    path = get_member_leaves_path(username, team_name)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump({k: list(v) for k, v in data.items()}, f, indent=4)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Warning: failed to save member_leaves.json: {e}")

def add_member_leave_exception(username: str, team_name: str, date_str: str, enrollment_id: str):
    """Add approved leave for enrollment_id on date_str."""
    data = load_member_leaves(username, team_name)
    arr = data.get(date_str, [])
    if enrollment_id not in arr:
        arr.append(enrollment_id)
    data[date_str] = arr
    save_member_leaves(username, team_name, data)

def remove_member_leave_exception(username: str, team_name: str, date_str: str, enrollment_id: str):
    """Remove a specific member leave exception if present."""
    data = load_member_leaves(username, team_name)
    if date_str in data:
        if enrollment_id in data[date_str]:
            data[date_str].remove(enrollment_id)
        if not data[date_str]:
            del data[date_str]
        save_member_leaves(username, team_name, data)





# @app.route('/api/enroll_capture', methods=['POST'])
# @login_required
# def api_enroll_capture():
#     username = session['username']
#     team_name = request.form['team']
#     member_name = request.form['name']
#     enroll_id = request.form['id']
#     contact = request.form['contact']
#     update_mode = request.form.get('update_mode', 'false').lower() == 'true'
#     team_path = get_team_path(username, team_name)
#     members_file = os.path.join(team_path, 'team_members.json')
#     members_data = json.load(open(members_file)) if os.path.exists(members_file) else {}
#
#     if enroll_id in members_data and not update_mode:
#         return jsonify({"status": "exists",
#                         "message": f"Enrollment ID '{enroll_id}' already exists for {members_data[enroll_id]['name']}. Do you want to update their details and recapture images?"})
#
#     dataset_path = get_team_data_path(username, team_name, 'dataset')
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path, exist_ok=True)
#
#     if enroll_id in members_data and update_mode:
#         old_member_name = members_data[enroll_id].get('name', 'unknown')
#         old_member_dir_name = f"{enroll_id}_{old_member_name.replace(' ', '_')}"
#         old_member_path = os.path.join(dataset_path, old_member_dir_name)
#         if os.path.exists(old_member_path):
#             shutil.rmtree(old_member_path)
#
#     member_dir_name = f"{enroll_id}_{member_name.replace(' ', '_')}"
#     member_path = os.path.join(dataset_path, member_dir_name)
#     os.makedirs(member_path, exist_ok=True)
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         shutil.rmtree(member_path, ignore_errors=True)
#         return jsonify({"status": "error", "message": "Could not access the camera."})
#
#     count, IMG_GOAL, start_time = 0, 50, time.time()
#     while count < IMG_GOAL and time.time() - start_time < 25:
#         ret, frame = cap.read()
#         if not ret: continue
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
#         if len(faces) == 1:
#             (x, y, w, h) = faces[0]
#             cv2.imwrite(os.path.join(member_path, f'{count}.jpg'), frame[y:y + h, x:x + w])
#             count += 1
#         time.sleep(0.1)
#     cap.release()
#
#     if count < 20:
#         shutil.rmtree(member_path, ignore_errors=True)
#         return jsonify({"status": "error", "message": f"Capture failed. Only got {count}/{IMG_GOAL} images."})
#
#     if update_mode:
#         members_data[enroll_id]['name'] = member_name
#         members_data[enroll_id]['contact'] = contact
#         if 'additional_data' not in members_data[enroll_id]:
#             members_data[enroll_id]['additional_data'] = {}
#         message = f"Successfully updated {member_name} with {count} new images."
#     else:
#         members_data[enroll_id] = {
#             'name': member_name, 'enrollment_id': enroll_id, 'contact': contact,
#             'enrollment_date': datetime.now().strftime('%Y-%m-%d'),
#             'status': 'active', 'leaving_date': None,
#             'additional_data': {}
#         }
#         message = f"{count} images captured for new member {member_name}."
#
#     with open(members_file, 'w') as f:
#         json.dump(members_data, f, indent=4)
#     return jsonify({"status": "success", "message": message})
#


# new capture function
# from deepface import DeepFace  # Ensure this import is at the top of your file


@app.route('/api/enroll_capture', methods=['POST'])
@login_required
def api_enroll_capture():
    print("\n--- STARTING ENROLLMENT PROCESS ---")
    try:
        username = session['username']
        team_name = request.form.get('team')
        member_name = request.form.get('name')
        enroll_id = request.form.get('id')
        contact = request.form.get('contact')
        update_mode = request.form.get('update_mode', 'false').lower() == 'true'

        print(f"[INFO] Leader: {username}, Team: {team_name}, Member: {member_name}")

        if not team_name:
            print("[ERROR] Team name is missing from the request!")
            return jsonify({"status": "error", "message": "Team name was not provided."})

        team_path = get_team_path(username, team_name)
        members_file = os.path.join(team_path, 'team_members.json')
        encodings_file = os.path.join(team_path, 'encodings.json')

        print(f"[INFO] Determined team path: {team_path}")

        members_data = read_json_file(members_file, {})
        if enroll_id in members_data and not update_mode:
            return jsonify({"status": "exists",
                            "message": f"Enrollment ID '{enroll_id}' already exists for {members_data[enroll_id]['name']}. Do you want to update their details and recapture images?"})

        encodings_data = read_json_file(encodings_file, {})
        member_encodings = []

        # (Image capture logic remains the same)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return jsonify({"status": "error", "message": "Could not access camera."})
        count, IMG_GOAL, start_time = 0, 20, time.time()
        while count < IMG_GOAL and time.time() - start_time < 20:
            ret, frame = cap.read();
            if not ret: continue;
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector(rgb, 1)
            if len(boxes) == 1:
                face_img = frame[boxes[0].top():boxes[0].bottom(), boxes[0].left():boxes[0].right()]
                try:
                    embedding_obj = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=False)
                    member_encodings.append(embedding_obj[0]["embedding"])
                    count += 1
                except:
                    pass
            time.sleep(0.2)
        cap.release()

        if count < 10: return jsonify({"status": "error", "message": f"Capture failed. Only got {count} clear images."})

        if update_mode and enroll_id in members_data:
            members_data[enroll_id]['name'] = member_name;
            members_data[enroll_id]['contact'] = contact
            message = f"Successfully updated {member_name}."
        else:
            members_data[enroll_id] = {'name': member_name, 'enrollment_id': enroll_id, 'contact': contact,
                                       'enrollment_date': datetime.now().strftime('%Y-%m-%d'), 'status': 'active',
                                       'additional_data': {}}
            message = f"Captured images for new member {member_name}."

        encodings_data[enroll_id] = member_encodings

        print(f"[ACTION] Attempting to save member details to: {members_file}")
        write_json_file(members_file, members_data)
        print(f"[ACTION] Attempting to save embeddings to: {encodings_file}")
        write_json_file(encodings_file, encodings_data)

        print("--- ENROLLMENT PROCESS COMPLETED SUCCESSFULLY ---\n")
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        print(f"[CRITICAL ERROR] Enrollment failed: {e}")
        return jsonify({"status": "error", "message": "A server error occurred during enrollment."}), 500



@app.route('/api/get_team_fields/<path:team_name>')
def api_get_team_fields(team_name: str):
    """
    API to get the dynamic form fields for a given team.
    This version works for both logged-in leaders and members.
    """
    # Check for leader or member session to get the correct leader username
    if 'username' in session:
        username = session['username']
    elif 'member_info' in session:
        username = session['member_info']['leader_username']
    else:
        # If no valid session is found, deny access
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    fields = get_team_fields_config(username, team_name)
    return jsonify(fields)
# --- NEW API to save custom fields ---
@app.route('/api/save_team_fields/<path:team_name>', methods=['POST'])
@login_required
def api_save_team_fields(team_name: str):
    """Saves the custom field configuration for a team."""
    username = session['username']
    new_fields_from_client = request.json.get('fields', [])
    if not all('key' in f and 'label' in f for f in new_fields_from_client):
        return jsonify({"status": "error", "message": "Invalid field format."})

    permanent_fields = [f for f in get_team_fields_config(username, team_name) if f.get('permanent')]

    final_fields = permanent_fields + new_fields_from_client

    config_path = os.path.join(get_team_path(username, team_name), 'team_config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump(final_fields, f, indent=4)
        return jsonify({"status": "success", "message": "Custom fields saved successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to save fields: {str(e)}"})

@app.route('/api/update_member_details', methods=['POST'])
@login_required
def api_update_member_details():
    """API to save or update the additional (dynamic) details for a member."""
    username = session['username']
    data = request.json
    team_name = data.get('team_name')
    enroll_id = data.get('enrollment_id')
    additional_data = data.get('additional_data', {})
    if not all([team_name, enroll_id]):
        return jsonify({"status": "error", "message": "Missing team or enrollment ID."})
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if not os.path.exists(members_file):
        return jsonify({"status": "error", "message": "Team members file not found."})
    with open(members_file, 'r+') as f:
        members_data = json.load(f)
        if enroll_id in members_data:
            members_data[enroll_id]['additional_data'] = additional_data
            f.seek(0)
            f.truncate()
            json.dump(members_data, f, indent=4)
            return jsonify({"status": "success", "message": "Additional details saved."})
        else:
            return jsonify({"status": "error", "message": "Member not found."})

@app.route('/api/remove_member', methods=['POST'])
@login_required
def api_remove_member():
    """API to mark a member as inactive."""
    username = session['username']
    team_name = request.json.get('team_name')
    enroll_id = request.json.get('enrollment_id')
    if not all([team_name, enroll_id]):
        return jsonify({"status": "error", "message": "Missing data."})
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if not os.path.exists(members_file):
        return jsonify({"status": "error", "message": "Team data not found."})
    with open(members_file, 'r+') as f:
        members_data = json.load(f)
        if enroll_id in members_data and members_data[enroll_id]['status'] == 'active':
            members_data[enroll_id]['status'] = 'inactive'
            members_data[enroll_id]['leaving_date'] = datetime.now().strftime('%Y-%m-%d')
            f.seek(0)
            f.truncate()
            json.dump(members_data, f, indent=4)
            return jsonify({"status": "success", "message": f"Member {enroll_id} has been marked as inactive."})
        else:
            return jsonify({"status": "error", "message": "Member not found or already inactive."})

@app.route('/api/reactivate_member', methods=['POST'])
@login_required
def api_reactivate_member():
    """API to reactivate a previously inactive member."""
    username = session['username']
    team_name = request.json.get('team_name')
    enroll_id = request.json.get('enrollment_id')
    if not all([team_name, enroll_id]):
        return jsonify({"status": "error", "message": "Missing data."})
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if not os.path.exists(members_file):
        return jsonify({"status": "error", "message": "Team data not found."})
    with open(members_file, 'r+') as f:
        members_data = json.load(f)
        if enroll_id in members_data and members_data[enroll_id]['status'] == 'inactive':
            members_data[enroll_id]['status'] = 'active'
            members_data[enroll_id]['leaving_date'] = None
            f.seek(0)
            f.truncate()
            json.dump(members_data, f, indent=4)
            return jsonify({"status": "success", "message": f"Member {enroll_id} has been reactivated."})
        else:
            return jsonify({"status": "error", "message": "Member not found or is already active."})

@app.route('/api/delete_member_permanently', methods=['POST'])
@login_required
def api_delete_member_permanently():
    username = session['username']
    team_name, enroll_id = request.json.get('team_name'), request.json.get('enrollment_id')
    if not all([team_name, enroll_id]):
        return jsonify({"status": "error", "message": "Missing required data."})
    dataset_path = get_team_data_path(username, team_name, 'dataset')
    if os.path.exists(dataset_path):
        folder_to_delete = next(
            (os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.startswith(f"{enroll_id}_")), None)
        if folder_to_delete:
            try:
                shutil.rmtree(folder_to_delete)
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error deleting image folder: {e}"})
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if os.path.exists(members_file):
        with open(members_file, 'r+') as f:
            members_data = json.load(f)
            if enroll_id in members_data:
                members_data[enroll_id]['status'] = 'deleted'
                members_data[enroll_id]['leaving_date'] = datetime.now().strftime('%Y-%m-%d')
                f.seek(0);
                f.truncate()
                json.dump(members_data, f, indent=4)
                return jsonify({"status": "success",
                                "message": f"Image data for member {enroll_id} deleted. Attendance history retained."})
    return jsonify({"status": "error", "message": "Member not found."})

@app.route('/api/delete_enrollment', methods=['POST'])
@login_required
def api_delete_enrollment():
    username = session['username']
    team_name, enroll_id = request.json.get('team'), request.json.get('id')
    if not all([team_name, enroll_id]):
        return jsonify({"status": "error", "message": "Missing team name or ID."})
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if not os.path.exists(members_file):
        return jsonify({"status": "error", "message": "Member file not found."})
    with open(members_file, 'r') as f:
        members_data = json.load(f)
    if enroll_id not in members_data:
        return jsonify({"status": "error", "message": "Enrollment ID not found in records."})
    member_name_to_delete = members_data[enroll_id].get('name', 'unknown')
    dir_name_to_delete = f"{enroll_id}_{member_name_to_delete.replace(' ', '_')}"
    dataset_path = get_team_data_path(username, team_name, 'dataset')
    folder_to_delete = os.path.join(dataset_path, dir_name_to_delete)
    if os.path.exists(folder_to_delete):
        shutil.rmtree(folder_to_delete, ignore_errors=True)
    del members_data[enroll_id]
    with open(members_file, 'w') as f:
        json.dump(members_data, f, indent=4)
    return jsonify({"status": "success", "message": f"All data for Enrollment ID {enroll_id} has been permanently deleted."})

# --- Leave single-day endpoints (preserved but adjusted route types where used by front-end) ---

@app.route('/api/mark_leave_day', methods=['POST'])
@login_required
def api_mark_leave_day():
    username = session['username']
    team_name = request.json.get('team_name')
    date_str = request.json.get('date')
    if not all([team_name, date_str]):
        return jsonify({"status": "error", "message": "Missing team name or date."})
    team_path = get_team_path(username, team_name)
    leave_file = os.path.join(team_path, 'leave_dates.json')
    leave_dates = json.load(open(leave_file)) if os.path.exists(leave_file) else []
    if date_str not in leave_dates:
        leave_dates.append(date_str)
        with open(leave_file, 'w') as f:
            json.dump(sorted(leave_dates), f, indent=4)
        return jsonify({"status": "success", "message": f"{date_str} has been marked as a leave day for {team_name}."})
    else:
        return jsonify({"status": "info", "message": "This date is already marked as a leave day."})

@app.route('/api/get_team_leave_dates/<path:team_name>')
@login_required
def api_get_team_leave_dates(team_name):
    username = session['username']
    team_path = get_team_path(username, team_name)
    leave_file = os.path.join(team_path, 'leave_dates.json')
    if os.path.exists(leave_file):
        with open(leave_file, 'r') as f:
            leave_dates = json.load(f)
        return jsonify(leave_dates)
    return jsonify([])

@app.route('/api/delete_leave_day', methods=['POST'])
@login_required
def api_delete_leave_day():
    """
    Delete leave day for a particular team.
    Additionally, we record a team-specific cancellation note in leave_cancellations.json
    so the calendar UI can display "for team X cancel leave" notes if the global tree remains (other teams still have that day).
    """
    username = session['username']
    team_name = request.json.get('team_name')
    date_str = request.json.get('date')
    if not all([team_name, date_str]):
        return jsonify({"status": "error", "message": "Missing team name or date."})
    team_path = get_team_path(username, team_name)
    leave_file = os.path.join(team_path, 'leave_dates.json')
    if not os.path.exists(leave_file):
        return jsonify({"status": "error", "message": "No leave days file found for this team."})
    with open(leave_file, 'r') as f:
        leave_dates = json.load(f)
    if date_str in leave_dates:
        leave_dates.remove(date_str)
        with open(leave_file, 'w') as f:
            json.dump(sorted(leave_dates), f, indent=4)
        # record cancellation note (team-specific)
        canc = load_cancellations(username)
        canc.append({"date": date_str, "team": team_name, "note": f"For team {team_name} cancel leave"})
        save_cancellations(username, canc)
        return jsonify({"status": "success", "message": f"Leave day {date_str} has been cancelled for {team_name}."})
    else:
        # even if not in team leave list, still record a cancellation note (so UI can show it)
        canc = load_cancellations(username)
        canc.append({"date": date_str, "team": team_name, "note": f"For team {team_name} cancel leave (was not present)"})
        save_cancellations(username, canc)
        return jsonify({"status": "info", "message": "This date was not found in the leave schedule for this team. Cancellation note recorded."})

@app.route('/api/delete_leave_for_all_teams', methods=['POST'])
@login_required
def api_delete_leave_for_all_teams():
    """
    Delete leave day from all teams.
    This will remove the date across all team's leave_dates.json and remove any cancellation entries for that date.
    """
    username = session['username']
    date_str = request.json.get('date')
    if not date_str:
        return jsonify({"status": "error", "message": "Date is required."})
    teams_path = os.path.join(get_user_path(username), 'teams')
    if not os.path.exists(teams_path):
        return jsonify({"status": "info", "message": "No teams found to delete leave from."})
    teams = [d for d in os.listdir(teams_path) if os.path.isdir(os.path.join(teams_path, d))]
    teams_updated_count = 0
    for team_name in teams:
        team_path = get_team_path(username, team_name)
        leave_file = os.path.join(team_path, 'leave_dates.json')
        if os.path.exists(leave_file):
            with open(leave_file, 'r') as f:
                leave_dates = json.load(f)
            if date_str in leave_dates:
                leave_dates.remove(date_str)
                with open(leave_file, 'w') as f:
                    json.dump(sorted(leave_dates), f, indent=4)
                teams_updated_count += 1
    # remove cancellations for that date too
    canc = load_cancellations(username)
    new_canc = [c for c in canc if c.get('date') != date_str]
    if len(new_canc) != len(canc):
        save_cancellations(username, new_canc)
    if teams_updated_count > 0:
        return jsonify({"status": "success", "message": f"Leave day {date_str} has been cancelled for all your teams."})
    else:
        return jsonify({"status": "info", "message": f"The leave day {date_str} was not found in any of your teams' schedules."})





# --- Training model endpoint (preserved) ---
# @app.route('/api/train_model', methods=['POST'])
# @login_required
# def api_train_model():
#     username = session['username']
#     team_name = request.json.get('team')
#     dataset_path = get_team_data_path(username, team_name, 'dataset')
#     members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
#     if not os.path.exists(members_file):
#         return jsonify({"status": "error", "message": "No members in this team."})
#     with open(members_file, 'r') as f:
#         all_members = json.load(f)
#     encodings, ids = [], []
#     active_member_dirs = [d for d in os.listdir(dataset_path) if
#                           os.path.isdir(os.path.join(dataset_path, d)) and d.split('_')[0] in all_members and
#                           all_members[d.split('_')[0]]['status'] == 'active']
#     if len(active_member_dirs) < 2:
#         return jsonify({"status": "error", "message": "Training requires at least two ACTIVE members with image data."})
#     for member_dir in active_member_dirs:
#         enroll_id = member_dir.split('_')[0]
#         person_path = os.path.join(dataset_path, member_dir)
#         for img_name in os.listdir(person_path):
#             try:
#                 image = cv2.imread(os.path.join(person_path, img_name))
#                 if image is None:
#                     continue
#                 rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 boxes = detector(rgb, 1)
#                 if len(boxes) != 1:
#                     continue
#                 shape = sp(rgb, boxes[0])
#                 encoding = np.array(facerec.compute_face_descriptor(rgb, shape))
#                 encodings.append(encoding)
#                 ids.append(enroll_id)
#             except Exception as e:
#                 print(f"Skipping image {img_name} in {member_dir} due to error: {e}")
#                 continue
#     if not encodings or len(set(ids)) < 2:
#         return jsonify({"status": "error",
#                         "message": "Could not generate enough valid face encodings for training. Please ensure members have high-quality images."})
#     le = LabelEncoder()
#     labels = le.fit_transform(ids)
#     n_neighbors = min(len(set(labels)), 5)
#     classifier = KNeighborsClassifier(n_neighbors=n_neighbors if n_neighbors > 0 else 1, weights='distance',
#                                       metric='euclidean')
#     classifier.fit(encodings, labels)
#     team_path = get_team_path(username, team_name)
#     with open(os.path.join(team_path, 'classifier.pkl'), 'wb') as f:
#         pickle.dump(classifier, f)
#     with open(os.path.join(team_path, 'label_encoder.pkl'), 'wb') as f:
#         pickle.dump(le, f)
#     return jsonify({"status": "success",
#                     "message": f"Model for '{team_name}' trained successfully on {len(set(ids))} active members."})


#new train model

from sklearn.svm import SVC  # Ensure this import is at the top of your file


@app.route('/api/train_model', methods=['POST'])
@login_required
def api_train_model():
    print("\n--- STARTING TRAINING PROCESS ---")
    try:
        username = session['username']
        team_name = request.json.get('team')

        print(f"[INFO] Leader: {username}, Team: {team_name}")

        if not team_name:
            print("[ERROR] Team name is missing from the request!")
            return jsonify({"status": "error", "message": "Team name was not provided."})

        team_path = get_team_path(username, team_name)
        encodings_file = os.path.join(team_path, 'encodings.json')
        members_file = os.path.join(team_path, 'team_members.json')

        print(f"[INFO] Determined team path: {team_path}")
        print(f"[INFO] Reading data from {encodings_file} and {members_file}")

        if not os.path.exists(encodings_file) or not os.path.exists(members_file):
            return jsonify({"status": "error", "message": "No member data or embeddings found to train."})

        all_encodings = read_json_file(encodings_file, {})
        all_members = read_json_file(members_file, {})

        # (Logic to prepare data remains the same)
        encodings, labels = [], []
        for enroll_id, member_data in all_members.items():
            if member_data.get('status') == 'active' and enroll_id in all_encodings:
                for encoding in all_encodings[enroll_id]:
                    encodings.append(encoding);
                    labels.append(enroll_id)

        if len(set(labels)) < 2:
            return jsonify(
                {"status": "error", "message": "Training requires at least two active members with captured images."})

        le = LabelEncoder();
        labels_encoded = le.fit_transform(labels)
        classifier = SVC(kernel='rbf', probability=True, gamma='auto')
        classifier.fit(encodings, labels_encoded)

        classifier_path = os.path.join(team_path, 'classifier.pkl')
        encoder_path = os.path.join(team_path, 'label_encoder.pkl')

        print(f"[ACTION] Attempting to save classifier to: {classifier_path}")
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        print(f"[ACTION] Attempting to save label encoder to: {encoder_path}")
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)

        print("--- TRAINING PROCESS COMPLETED SUCCESSFULLY ---\n")
        return jsonify({"status": "success",
                        "message": f"Model for '{team_name}' trained successfully on {len(set(labels))} active members."})
    except Exception as e:
        print(f"[CRITICAL ERROR] Training failed: {e}")
        return jsonify({"status": "error", "message": "A server error occurred during training."}), 500


# --- Today's/Final attendance and rules (preserved) ---
# @app.route('/api/get_todays_attendance/<path:team_name>')
# @login_required
# def api_get_todays_attendance(team_name):
#     username = session['username']
#     members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
#     if not os.path.exists(members_file):
#         return jsonify({'morning': [], 'evening': []})
#     all_members = json.load(open(members_file))
#     log_file = os.path.join(get_team_data_path(username, team_name, 'attendance_logs'),
#                             f"{datetime.now().strftime('%Y-%m-%d')}.json")
#     today_logs = json.load(open(log_file)) if os.path.exists(log_file) else {}
#     morning_attendance, evening_attendance = [], []
#     thirteen_pm = dt_time(13, 0)
#     for enroll_id, member_info in all_members.items():
#         if member_info['status'] == 'active' and enroll_id in today_logs:
#             scans = sorted([datetime.strptime(t, '%H:%M:%S').time() for t in today_logs[enroll_id]])
#             morning_scan = next((s for s in scans if s < thirteen_pm), None)
#             evening_scan = next((s for s in reversed(scans) if s >= thirteen_pm), None)
#             if morning_scan:
#                 morning_attendance.append({"name": member_info['name'], "time": morning_scan.strftime('%H:%M:%S')})
#             if evening_scan:
#                 evening_attendance.append({"name": member_info['name'], "time": evening_scan.strftime('%H:%M:%S')})
#     return jsonify({'morning': sorted(morning_attendance, key=lambda x: x['time']),
#                     'evening': sorted(evening_attendance, key=lambda x: x['time'])})

#new one
@app.route('/api/get_todays_attendance/<path:team_name>')
@login_required
def api_get_todays_attendance(team_name):
    """
    Provides a live, calculated, final-report-style summary for all active
    members for the current day.
    """
    username = session['username']
    today = datetime.now().date()

    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    all_members = read_json_file(members_file, {})

    todays_report = []
    for enroll_id, member_info in all_members.items():
        # We only need to generate a report for active members
        if member_info.get('status') == 'active':
            # Reuse the central calculation engine for perfect consistency
            member_daily_records = _calculate_final_attendance_for_member(username, team_name, enroll_id, today, today)
            if member_daily_records:
                # The engine returns a list, but for a single day, we just need the first item
                todays_report.append(member_daily_records[0])

    # Sort the report by name for a consistent display
    todays_report.sort(key=lambda x: x.get('name', ''))
    return jsonify(todays_report)









def get_default_attendance_rules():
    return {
        "criteria_type": "time",
        "office_start_time": "09:30",
        "half_day": {"by_time": {"in_time_safe_range_end": "10:00", "in_time_late_range_end": "11:00",
                                 "required_out_time": "14:00"}, "by_hours": {"min_hours": 4.0}},
        "full_day": {"by_time": {"required_out_time_early_go": "17:30", "required_out_time_present": "18:30"},
                     "by_hours": {"min_hours_early_go": 7.0, "min_hours_present": 8.0}}
    }

@app.route('/api/get_attendance_rules/<path:team_name>')
@login_required
def api_get_attendance_rules(team_name):
    username = session['username']
    rules_file = os.path.join(get_team_path(username, team_name), 'attendance_rules.json')
    if os.path.exists(rules_file):
        with open(rules_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify(get_default_attendance_rules())

@app.route('/api/save_attendance_rules/<path:team_name>', methods=['POST'])
@login_required
def api_save_attendance_rules(team_name):
    username = session['username']
    rules_data = request.json
    rules_file = os.path.join(get_team_path(username, team_name), 'attendance_rules.json')
    try:
        with open(rules_file, 'w') as f:
            json.dump(rules_data, f, indent=4)
        return jsonify({"status": "success", "message": "Attendance rules saved successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to save rules: {str(e)}"})









# @app.route('/api/get_final_attendance/<team_name>/<date_str>')
# @login_required
# def api_get_final_attendance(team_name, date_str):
#     username = session['username']
#     final_report = []
#     members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
#     if not os.path.exists(members_file):
#         return jsonify([])
#     with open(members_file, 'r') as f:
#         all_members = json.load(f)
#
#     # load rules (fallback to defaults)
#     rules_file = os.path.join(get_team_path(username, team_name), 'attendance_rules.json')
#     rules = {}
#     if os.path.exists(rules_file):
#         with open(rules_file, 'r') as f:
#             try:
#                 rules = json.load(f)
#             except Exception:
#                 rules = get_default_attendance_rules()
#     else:
#         rules = get_default_attendance_rules()
#
#     # team announced leave dates
#     leave_file = os.path.join(get_team_path(username, team_name), 'leave_dates.json')
#     leave_dates = []
#     if os.path.exists(leave_file):
#         try:
#             with open(leave_file, 'r') as f:
#                 leave_dates = json.load(f)
#         except Exception:
#             leave_dates = []
#
#     # logs for date
#     log_file = os.path.join(get_team_data_path(username, team_name, 'attendance_logs'), f"{date_str}.json")
#     date_logs = json.load(open(log_file)) if os.path.exists(log_file) else {}
#
#     report_date = None
#     try:
#         report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
#     except Exception:
#         report_date = None
#
#     for enroll_id, member_info in all_members.items():
#         report_item = {"name": member_info['name'], "enrollment_id": enroll_id, "time_in": None, "time_out": None,
#                        "working_hours": "0h 0m", "late_coming": False, "early_going": False, "status": "Absent"}
#         # joined/left checks
#         enrollment_date = None
#         try:
#             enrollment_date = datetime.strptime(member_info.get('enrollment_date', ''), '%Y-%m-%d').date()
#         except Exception:
#             enrollment_date = None
#
#         if report_date and enrollment_date and report_date < enrollment_date:
#             report_item['status'] = "Not Joined"
#             final_report.append(report_item)
#             continue
#
#         if date_str in leave_dates:
#             report_item['status'] = "Leave"
#             final_report.append(report_item)
#             continue
#
#         # If member is explicitly marked absent/inactive status handling will follow but we continue with logs
#         if enroll_id in date_logs and date_logs[enroll_id]:
#             scans_strlist = date_logs[enroll_id]
#             # parse times into datetime objects (today's date)
#             scans = []
#             for t in scans_strlist:
#                 try:
#                     scans.append(datetime.strptime(t, '%H:%M:%S'))
#                 except Exception:
#                     pass
#             scans = sorted(scans)
#             if not scans:
#                 report_item['status'] = "Absent"
#                 final_report.append(report_item)
#                 continue
#
#             # Determine morning/afternoon presence by your 13:00 rule
#             thirteen_pm = dt_time(13, 0)
#             morning_scan = any(s.time() < thirteen_pm for s in scans)
#             evening_scan = any(s.time() >= thirteen_pm for s in scans)
#
#             # If only ONE timestamp exists -> treat as only Time In and NO Time Out
#             if len(scans) == 1:
#                 report_item['time_in'] = scans[0].strftime('%H:%M:%S')
#                 report_item['time_out'] = '-'  # explicitly show dash instead of duplicating time_in
#                 report_item['working_hours'] = "0h 0m"
#
#                 # decide status for single scan:
#                 # if it is before 13:00 -> considered half-day (morning) else half-day (evening)
#                 # you can customize this mapping if you'd rather mark it Present/Half-Day differently
#                 if morning_scan or evening_scan:
#                     report_item['status'] = 'Half Day'
#                     # mark late if morning scan is after safe_end
#                     safe_end_str = rules.get('half_day', {}).get('by_time', {}).get('in_time_safe_range_end', '10:00')
#                     try:
#                         safe_end = datetime.strptime(safe_end_str, '%H:%M').time()
#                         if morning_scan and scans[0].time() > safe_end:
#                             report_item['late_coming'] = True
#                     except Exception:
#                         pass
#                 else:
#                     report_item['status'] = 'Absent'
#                 final_report.append(report_item)
#                 continue
#
#             # More than one scan -> normal behaviour: first = time_in, last = time_out
#             time_in_dt = scans[0]
#             time_out_dt = scans[-1]
#             report_item['time_in'] = time_in_dt.strftime('%H:%M:%S')
#             report_item['time_out'] = time_out_dt.strftime('%H:%M:%S')
#
#             # calculate working hours
#             work_delta = time_out_dt - time_in_dt
#             total_seconds = work_delta.total_seconds() if work_delta.total_seconds() >= 0 else 0
#             hours, remainder = divmod(total_seconds, 3600)
#             minutes, _ = divmod(remainder, 60)
#             report_item['working_hours'] = f"{int(hours)}h {int(minutes)}m"
#
#             # late / early logic (existing rules)
#             def to_time_or_none(tstr):
#                 try:
#                     return datetime.strptime(tstr, '%H:%M').time() if tstr else None
#                 except Exception:
#                     return None
#
#             safe_end = to_time_or_none(rules.get('half_day', {}).get('by_time', {}).get('in_time_safe_range_end', '10:00'))
#             hd_out = to_time_or_none(rules.get('half_day', {}).get('by_time', {}).get('required_out_time', '14:00'))
#             fd_early = to_time_or_none(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_early_go', '17:30'))
#             fd_full = to_time_or_none(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_present', '18:30'))
#
#             time_in = time_in_dt.time()
#             time_out = time_out_dt.time()
#             is_late = False
#             if safe_end:
#                 is_late = time_in > safe_end
#                 if is_late:
#                     report_item['late_coming'] = True
#
#             # decide status (mirror existing logic)
#             if rules.get('criteria_type') == 'hours':
#                 hd_h = rules.get('half_day', {}).get('by_hours', {}).get('min_hours', 4)
#                 fd_early_h = rules.get('full_day', {}).get('by_hours', {}).get('min_hours_early_go', 7)
#                 fd_full_h = rules.get('full_day', {}).get('by_hours', {}).get('min_hours_present', 8)
#                 work_hours_float = total_seconds / 3600.0
#                 if work_hours_float >= fd_full_h:
#                     report_item['status'] = 'Present'
#                 elif work_hours_float >= fd_early_h:
#                     report_item['status'] = 'Present (Early Go)'
#                     report_item['early_going'] = True
#                 elif work_hours_float >= hd_h:
#                     report_item['status'] = 'Half Day'
#                 else:
#                     report_item['status'] = 'Absent'
#             else:
#                 if time_out >= fd_full:
#                     report_item['status'] = 'Present (Late)' if is_late else 'Present'
#                 elif time_out >= fd_early:
#                     report_item['status'] = 'Present (Late, Early Go)' if is_late else 'Present (Early Go)'
#                     report_item['early_going'] = True
#                 elif time_out >= hd_out:
#                     report_item['status'] = 'Half Day (Late)' if is_late else 'Half Day'
#                 else:
#                     report_item['status'] = 'Absent'
#
#         else:
#             # no logs -> Absent (unless leave/other earlier checks)
#             report_item['status'] = 'Absent'
#
#         final_report.append(report_item)
#
#     return jsonify(final_report)






# @app.route('/api/get_team_location/<team_name>')
# @login_required
# def get_team_location(team_name):
#     ...
#
# @app.route('/api/get_member_final_attendance/<team_name>/<enrollment_id>')
# @login_required
# def get_member_final_attendance(team_name, enrollment_id):
#     start = request.args.get('start')
#     end = request.args.get('end')
#     # fetch data from DB and return JSON
#     ...







def read_member_leaves(username: str, team_name: str, date_str: str = None):
    """
    Read member-specific leave records.
    File: <team_path>/member_leaves.json

    Supported structures:
    A) { "2025-09-28": ["0103AL231226", "0103AL231227"], ... }
    B) { "2025-09-28": { "0103AL231226": {...}, ... }, ... }

    If date_str provided, returns the value for that date (list or dict) or empty structure.
    """
    try:
        team_path = get_team_path(username, team_name)
        ml_file = os.path.join(team_path, 'member_leaves.json')
        if not os.path.exists(ml_file):
            return {} if date_str else {}
        with open(ml_file, 'r') as f:
            data = json.load(f)
        if date_str:
            return data.get(date_str, {})
        return data
    except Exception as e:
        app.logger.error(f"read_member_leaves error for {username}/{team_name}: {e}")
        return {}






# new engine for final attendance
# ADD THIS NEW HELPER FUNCTION
def _calculate_final_attendance_for_member(username, team_name, enroll_id, start_date, end_date):
    """
    This is the central 'calculation engine'.
    UPDATED: Out-time is now locked after the second scan of the day.
    """
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    members_data = read_json_file(members_file, {})
    member_obj = members_data.get(enroll_id, {})
    member_name = member_obj.get('name', '')

    rules = read_json_file(os.path.join(get_team_path(username, team_name), 'attendance_rules.json'),
                           get_default_attendance_rules())
    team_leaves = read_json_file(os.path.join(get_team_path(username, team_name), 'leave_dates.json'), [])
    member_leaves = read_json_file(os.path.join(get_team_path(username, team_name), 'member_leaves.json'), {})

    try:
        enrollment_date = datetime.strptime(member_obj.get('enrollment_date'), '%Y-%m-%d').date() if member_obj.get(
            'enrollment_date') else None
        leaving_date = datetime.strptime(member_obj.get('leaving_date'), '%Y-%m-%d').date() if member_obj.get(
            'leaving_date') else None
    except (ValueError, TypeError):
        enrollment_date, leaving_date = None, None

    records = []
    today = datetime.now().date()
    date_iter = start_date

    while date_iter <= end_date:
        date_str = date_iter.strftime('%Y-%m-%d')
        report_item = {'date': date_str, 'enrollment_id': enroll_id, 'name': member_name, 'time_in': '-',
                       'time_out': '-', 'working_hours': '0h 0m', 'late_coming': False, 'early_going': False,
                       'status': 'Absent'}

        if date_iter > today or \
                (enrollment_date and date_iter < enrollment_date) or \
                (leaving_date and date_iter > leaving_date):
            report_item['status'] = 'N/A'
        elif date_str in team_leaves or (date_str in member_leaves and enroll_id in member_leaves.get(date_str, [])):
            report_item['status'] = 'Leave'
        else:
            scans_file = os.path.join(get_team_data_path(username, team_name, 'attendance_logs'), f"{date_str}.json")
            scans_by_member = read_json_file(scans_file, {})
            scans_for_member = scans_by_member.get(enroll_id, [])
            if len(scans_for_member) >= 2:
                parsed = sorted([datetime.strptime(t, '%H:%M:%S') for t in scans_for_member])

                # --- MODIFIED LOGIC ---
                # In-Time is the first scan. Out-Time is locked to the second scan.
                time_in_dt = parsed[0]
                time_out_dt = parsed[1]
                # --- END MODIFICATION ---

                total_seconds = (time_out_dt - time_in_dt).total_seconds()
                hours, rem = divmod(total_seconds, 3600)
                minutes, _ = divmod(rem, 60)
                report_item.update(
                    {"time_in": time_in_dt.strftime('%H:%M:%S'), "time_out": time_out_dt.strftime('%H:%M:%S'),
                     "working_hours": f"{int(hours)}h {int(minutes)}m"})

                status, is_late, is_early = 'Absent', False, False
                criteria_type = rules.get('criteria_type', 'time')
                if criteria_type == 'hours':
                    work_hours_float = total_seconds / 3600.0
                    min_full = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_present', 8.0))
                    min_early = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_early_go', 7.0))
                    min_half = float(rules.get('half_day', {}).get('by_hours', {}).get('min_hours', 4.0))
                    if work_hours_float >= min_full:
                        status = 'Present'
                    elif work_hours_float >= min_early:
                        status = 'Present'; is_early = True
                    elif work_hours_float >= min_half:
                        status = 'Half Day'
                else:
                    def to_time(t_str):
                        try:
                            return dt_time.fromisoformat(t_str) if t_str else None
                        except:
                            return None

                    safe_end = to_time(rules.get('half_day', {}).get('by_time', {}).get('in_time_safe_range_end'))
                    hd_out = to_time(rules.get('half_day', {}).get('by_time', {}).get('required_out_time'))
                    fd_early = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_early_go'))
                    fd_full = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_present'))
                    if safe_end and time_in_dt.time() > safe_end: is_late = True
                    if fd_early and time_out_dt.time() < fd_early: is_early = True
                    if fd_full and time_out_dt.time() >= fd_full:
                        status = 'Present'
                    elif fd_early and time_out_dt.time() >= fd_early:
                        status = 'Present'
                    elif hd_out and time_out_dt.time() >= hd_out:
                        status = 'Half Day'

                report_item['status'] = status
                if status != 'Absent':
                    report_item['late_coming'] = is_late
                    report_item['early_going'] = is_early
            elif len(scans_for_member) == 1:
                report_item['time_in'] = scans_for_member[0]

        records.append(report_item)
        date_iter += timedelta(days=1)

    return records


# REPLACE this function
@app.route('/api/get_final_attendance/<team_name>/<date_str>')
@login_required
def api_get_final_attendance(team_name, date_str):
    """ API for the leader's final report table on the Mark Attendance page. """
    username = session['username']
    try:
        report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return jsonify([])

    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    all_members = read_json_file(members_file, {})

    final_report = []
    for enroll_id in all_members:
        # For each member, call the new engine for just that single day
        member_records = _calculate_final_attendance_for_member(username, team_name, enroll_id, report_date,
                                                                report_date)
        if member_records:  # The function returns a list, we just need the first item
            final_report.append(member_records[0])

    return jsonify(final_report)

























# --- Replacement: api_get_final_attendance with flags & member-leaves handling ---

# @app.route('/api/get_final_attendance/<team_name>/<date_str>')
# @login_required
# def api_get_final_attendance(team_name, date_str):
#     """
#     Return final attendance report for a team on a particular date.
#     This function now checks:
#       - team-level announced leaves (leave_dates.json)
#       - member-specific leaves (member_leaves.json) -> take precedence as 'Leave' for that member
#       - attendance_flags (attendance_flags.json) -> if location_mismatch set for enroll_id on this date -> mark Absent
#       - attendance logs (attendance_logs/<date>.json)
#     Important policies implemented:
#       - If a member has only a single scan on a date -> treat as Absent (strict)
#       - If flags mark location_mismatch -> Absent and skip further processing
#       - Not Joined days (before enrollment) are preserved
#     """
#     username = session['username']
#     final_report = []
#
#     # Load members
#     members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
#     if not os.path.exists(members_file):
#         return jsonify([])
#
#     try:
#         with open(members_file, 'r') as f:
#             all_members = json.load(f)
#     except Exception as e:
#         app.logger.error(f"Failed to load members for {username}/{team_name}: {e}")
#         return jsonify([])
#
#     # Load rules (or default)
#     rules_file = os.path.join(get_team_path(username, team_name), 'attendance_rules.json')
#     rules = {}
#     if os.path.exists(rules_file):
#         try:
#             with open(rules_file, 'r') as f:
#                 rules = json.load(f)
#         except Exception as e:
#             app.logger.error(f"Failed to load rules for {username}/{team_name}: {e}")
#             rules = {}
#     else:
#         rules = get_default_attendance_rules()
#
#     # Team-level announced leave dates
#     leave_file = os.path.join(get_team_path(username, team_name), 'leave_dates.json')
#     leave_dates = []
#     if os.path.exists(leave_file):
#         try:
#             with open(leave_file, 'r') as f:
#                 leave_dates = json.load(f)
#         except Exception as e:
#             app.logger.error(f"Failed to read leave_dates for {username}/{team_name}: {e}")
#             leave_dates = []
#
#     # Member-specific leaves for the date (member_leaves.json)
#     member_leaves_for_date = read_member_leaves(username, team_name, date_str)  # could be list or dict
#
#     # Attendance logs for that date
#     log_file = os.path.join(get_team_data_path(username, team_name, 'attendance_logs'), f"{date_str}.json")
#     date_logs = {}
#     if os.path.exists(log_file):
#         try:
#             with open(log_file, 'r') as f:
#                 date_logs = json.load(f)
#         except Exception as e:
#             app.logger.error(f"Failed to read logs {log_file}: {e}")
#             date_logs = {}
#
#     # Parse date for comparisons
#     try:
#         report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
#     except Exception as e:
#         app.logger.error(f"Invalid date_str passed to api_get_final_attendance: {date_str}")
#         return jsonify([])
#
#     # Prepare rule-derived times if time-based
#     def to_time(t_str):
#         try:
#             return datetime.strptime(t_str, '%H:%M').time() if t_str else None
#         except:
#             return None
#
#     criteria_type = rules.get('criteria_type', 'time')
#     safe_end = to_time(rules.get('half_day', {}).get('by_time', {}).get('in_time_safe_range_end', '10:00'))
#     late_end = to_time(rules.get('half_day', {}).get('by_time', {}).get('in_time_late_range_end', '11:00'))
#     hd_out = to_time(rules.get('half_day', {}).get('by_time', {}).get('required_out_time', '14:00'))
#     fd_early = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_early_go', '17:30'))
#     fd_full = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_present', '18:30'))
#
#     # Pre-load flags for the date (attendance_flags.json)
#     flags_for_date = read_attendance_flags(username, team_name, date_str)
#
#     # Iterate members
#     for enroll_id, member_info in all_members.items():
#         report_item = {
#             "name": member_info.get('name', ''),
#             "enrollment_id": enroll_id,
#             "time_in": None,
#             "time_out": None,
#             "working_hours": "0h 0m",
#             "late_coming": False,
#             "early_going": False,
#             "status": "Absent"
#         }
#
#         # Enrollment date check
#         try:
#             enrollment_date = datetime.strptime(member_info.get('enrollment_date', '1970-01-01'), '%Y-%m-%d').date()
#         except:
#             enrollment_date = datetime(1970, 1, 1).date()
#
#         # If report_date is before joining -> Not Joined
#         if report_date < enrollment_date:
#             report_item['status'] = "Not Joined"
#             final_report.append(report_item)
#             continue
#
#         # 1) Member-level leave override (highest precedence for that member)
#         # member_leaves_for_date may be a list of enroll_ids or dict mapping enroll_id->meta
#         if member_leaves_for_date:
#             try:
#                 if isinstance(member_leaves_for_date, list) and enroll_id in member_leaves_for_date:
#                     report_item['status'] = "Leave"
#                     final_report.append(report_item)
#                     continue
#                 if isinstance(member_leaves_for_date, dict) and enroll_id in member_leaves_for_date:
#                     report_item['status'] = "Leave"
#                     final_report.append(report_item)
#                     continue
#             except Exception as e:
#                 app.logger.debug(f"member_leaves_for_date check error: {e}")
#
#         # 2) Flags override (e.g. location_mismatch) -> mark Absent and skip further logic
#         if flags_for_date and isinstance(flags_for_date, dict) and enroll_id in flags_for_date:
#             member_flag = flags_for_date.get(enroll_id, {})
#             if member_flag and isinstance(member_flag, dict) and member_flag.get('location_mismatch'):
#                 # Strict absent due to location mismatch (clear late/early)
#                 report_item['status'] = 'Absent'
#                 report_item['late_coming'] = False
#                 report_item['early_going'] = False
#                 final_report.append(report_item)
#                 continue
#
#         # 3) Team-level leave (announced) -> mark Leave for all members
#         if date_str in leave_dates:
#             report_item['status'] = "Leave"
#             final_report.append(report_item)
#             continue
#
#         # 4) If attendance logs exist for this enroll_id, analyze scans
#         if enroll_id in date_logs and date_logs[enroll_id]:
#             try:
#                 scans = sorted([datetime.strptime(t, '%H:%M:%S') for t in date_logs[enroll_id]])
#             except Exception as e:
#                 app.logger.error(f"Error parsing scans for {enroll_id} on {date_str}: {e}")
#                 scans = []
#
#             if not scans:
#                 # no scans -> Absent (default)
#                 report_item['status'] = "Absent"
#                 final_report.append(report_item)
#                 continue
#
#             # SINGLE SCAN POLICY: Strictly Absent
#             if len(scans) == 1:
#                 # Per your requirement: 1 scan -> treat as Absent (no late/early)
#                 report_item['time_in'] = scans[0].strftime('%H:%M:%S')
#                 report_item['time_out'] = '-'  # optional, keep as single-scan sentinel
#                 report_item['working_hours'] = "0h 0m"
#                 report_item['status'] = "Absent"
#                 report_item['late_coming'] = False
#                 report_item['early_going'] = False
#                 final_report.append(report_item)
#                 continue
#
#             # Normal multi-scan processing
#             time_in_dt = scans[0]
#             time_out_dt = scans[-1]
#             report_item['time_in'] = time_in_dt.strftime('%H:%M:%S')
#             report_item['time_out'] = time_out_dt.strftime('%H:%M:%S')
#
#             work_delta = time_out_dt - time_in_dt
#             total_seconds = max(0, work_delta.total_seconds())
#             hours, remainder = divmod(total_seconds, 3600)
#             minutes, _ = divmod(remainder, 60)
#             report_item['working_hours'] = f"{int(hours)}h {int(minutes)}m"
#
#             # By-hours rules
#             if criteria_type == 'hours':
#                 try:
#                     hd_h = float(rules.get('half_day', {}).get('by_hours', {}).get('min_hours', 4.0))
#                 except:
#                     hd_h = 4.0
#                 try:
#                     fd_early_h = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_early_go', 7.0))
#                 except:
#                     fd_early_h = 7.0
#                 try:
#                     fd_full_h = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_present', 8.0))
#                 except:
#                     fd_full_h = 8.0
#
#                 work_hours_float = total_seconds / 3600.0
#                 if work_hours_float >= fd_full_h:
#                     report_item['status'] = 'Present'
#                 elif work_hours_float >= fd_early_h:
#                     report_item['status'] = 'Present (Early Go)'
#                     report_item['early_going'] = True
#                 elif work_hours_float >= hd_h:
#                     report_item['status'] = 'Half Day'
#                 else:
#                     report_item['status'] = 'Absent'
#
#                 # Late detection by comparing time_in to safe_end (if provided)
#                 if safe_end and time_in_dt.time() > safe_end and report_item['status'] != 'Absent':
#                     report_item['late_coming'] = True
#
#             else:
#                 # Time-based rules
#                 is_late = False
#                 if safe_end and time_in_dt.time() > safe_end:
#                     is_late = True
#                     report_item['late_coming'] = True
#
#                 # If time_in after late_end => Absent
#                 if late_end and time_in_dt.time() > late_end:
#                     report_item['status'] = 'Absent'
#                 else:
#                     if time_out_dt.time() >= fd_full:
#                         report_item['status'] = 'Present (Late)' if is_late else 'Present'
#                     elif time_out_dt.time() >= fd_early:
#                         report_item['status'] = 'Present (Late, Early Go)' if is_late else 'Present (Early Go)'
#                         report_item['early_going'] = True
#                     elif time_out_dt.time() >= hd_out:
#                         report_item['status'] = 'Half Day (Late)' if is_late else 'Half Day'
#                     else:
#                         report_item['status'] = 'Absent'
#
#         else:
#             # No logs -> Absent (default)
#             report_item['status'] = "Absent"
#
#         final_report.append(report_item)
#
#     return jsonify(final_report)




@app.route('/api/capture_and_encode', methods=['POST'])
def api_capture_and_encode():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    decoded_image = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    if len(faces) != 1:
        return jsonify({"status": "error", "message": "Could not detect a single clear face. Please try again."})
    (x, y, w, h) = faces[0]
    shape = sp(rgb, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
    encoding = np.array(facerec.compute_face_descriptor(rgb, shape)).tolist()
    return jsonify({"status": "success", "encoding": encoding})

@app.route('/api/verify_face', methods=['POST'])
def api_verify_face():
    data = request.get_json()
    username = data['username']
    image_data = data['image'].split(',')[1]
    users = json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}
    if username not in users or 'face_encoding' not in users[username]:
        return jsonify({"status": "error", "message": "No face data found for this user."})
    stored_encoding = np.array(users[username]['face_encoding'])
    decoded_image = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    if len(faces) != 1:
        return jsonify({"status": "error", "message": "Could not detect a face."})
    (x, y, w, h) = faces[0]
    shape = sp(rgb, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
    current_encoding = np.array(facerec.compute_face_descriptor(rgb, shape))
    distance = np.linalg.norm(stored_encoding - current_encoding)
    if distance < 0.223:
        return jsonify({"status": "success", "message": "Face verified."})
    else:
        return jsonify({"status": "error", "message": "Face does not match."})

# @app.route('/api/verify_member_face', methods=['POST'])
# def api_verify_member_face():
#     data = request.get_json()
#     leader_username = data['leader_username']
#     team_name = data['team_name']
#     enrollment_id = data['enrollment_id']
#     image_data = data['image'].split(',')[1]
#     team_path = get_team_path(leader_username, team_name)
#     try:
#         with open(os.path.join(team_path, 'classifier.pkl'), 'rb') as f:
#             classifier = pickle.load(f)
#         with open(os.path.join(team_path, 'label_encoder.pkl'), 'rb') as f:
#             le = pickle.load(f)
#         with open(os.path.join(team_path, 'team_members.json'), 'r') as f:
#             members = json.load(f)
#     except FileNotFoundError:
#         return jsonify({"status": "error", "message": "Team model not trained or data is missing."})
#     decoded_image = base64.b64decode(image_data)
#     np_arr = np.frombuffer(decoded_image, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
#     if len(faces) != 1:
#         return jsonify({"status": "error", "message": "Could not detect a single face."})
#     (x, y, w, h) = faces[0]
#     shape = sp(rgb, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
#     encoding = np.array(facerec.compute_face_descriptor(rgb, shape)).reshape(1, -1)
#     preds = classifier.predict_proba(encoding)[0]
#     j = np.argmax(preds)
#     confidence = preds[j]
#     predicted_id = le.inverse_transform([j])[0]
#     if predicted_id == enrollment_id and confidence > 0.85:
#         session['member_info'] = members[enrollment_id]
#         session['member_info']['leader_username'] = leader_username
#         session['member_info']['team_name'] = team_name
#         return jsonify({"status": "success", "message": "Verification successful! Logging in..."})
#     else:
#         return jsonify({"status": "error", "message": "Face verification failed. Please try again."})

# new for this

@app.route('/api/verify_member_face', methods=['POST'])
def api_verify_member_face():
    data = request.get_json()
    leader_username = data['leader_username']
    team_name = data['team_name']
    enrollment_id = data['enrollment_id']
    image_data = data['image'].split(',')[1]

    team_path = get_team_path(leader_username, team_name)
    try:
        classifier = pickle.load(open(os.path.join(team_path, 'classifier.pkl'), 'rb'))
        le = pickle.load(open(os.path.join(team_path, 'label_encoder.pkl'), 'rb'))
        members = read_json_file(os.path.join(team_path, 'team_members.json'))
    except FileNotFoundError:
        return jsonify(
            {"status": "error", "message": "Team model not trained. Please ask your leader to train the model."})

    decoded_image = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        # UPGRADE: Use dlib to reliably find the face first
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detector(rgb_frame, 1)

        if len(boxes) != 1:
            return jsonify({"status": "error", "message": "Could not detect a single, clear face. Please try again."})

        # Pass the already-detected face to DeepFace
        face_img = rgb_frame[boxes[0].top():boxes[0].bottom(), boxes[0].left():boxes[0].right()]
        embedding_obj = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=False)
        embedding = embedding_obj[0]["embedding"]

        preds = classifier.predict_proba([embedding])[0]
        j = np.argmax(preds)
        confidence = preds[j]
        predicted_id = le.inverse_transform([j])[0]

        # Use a slightly lower confidence for login verification to be more forgiving
        if predicted_id == enrollment_id and confidence > 0.80:
            session['member_info'] = members[enrollment_id]
            session['member_info']['leader_username'] = leader_username
            session['member_info']['team_name'] = team_name
            return jsonify({"status": "success", "message": "Verification successful! Logging in..."})
        else:
            return jsonify({"status": "error", "message": "Face verification failed. Please try again."})

    except ValueError as e:
        print(f"DeepFace error during verification: {e}")
        return jsonify(
            {"status": "error", "message": "Could not process face. Please ensure the model is trained correctly."})
    except Exception as e:
        print(f"An unexpected error occurred during verification: {e}")
        return jsonify({"status": "error", "message": "An unexpected server error occurred."}), 500

@app.route('/api/get_member_attendance_history')
@member_login_required
def api_get_member_attendance_history():
    member_info = session['member_info']
    leader_username = member_info['leader_username']
    team_name = member_info['team_name']
    enrollment_id = member_info['enrollment_id']
    logs_path = get_team_data_path(leader_username, team_name, 'attendance_logs')
    full_history = []
    if not os.path.exists(logs_path):
        return jsonify([])
    team_path = get_team_path(leader_username, team_name)
    leave_file = os.path.join(team_path, 'leave_dates.json')
    leave_dates = json.load(open(leave_file)) if os.path.exists(leave_file) else []

    member_leaves = load_member_leaves(leader_username, team_name)

    enrollment_date = datetime.strptime(member_info['enrollment_date'], '%Y-%m-%d').date()
    end_date = datetime.now().date()
    all_logs = {}
    for log_filename in os.listdir(logs_path):
        if log_filename.endswith('.json'):
            date_str = log_filename.replace('.json', '')
            with open(os.path.join(logs_path, log_filename), 'r') as f:
                all_logs[date_str] = json.load(f)




    current_date = end_date
    while current_date >= enrollment_date:
        date_str = current_date.strftime('%Y-%m-%d')
        status, morning_time, evening_time = "A", "-", "-"

        # Member-level leave override
        if date_str in member_leaves and enrollment_id in member_leaves[date_str]:
            status = "Leave"

        else:
            if date_str in leave_dates:
                status = "Leave"

            else:
                date_logs = all_logs.get(date_str)
                if date_logs and enrollment_id in date_logs:
                    thirteen_pm = dt_time(13, 0)
                    scans = sorted([datetime.strptime(t, '%H:%M:%S').time() for t in date_logs[enrollment_id]])
                    morning_scan = next((s for s in scans if s < thirteen_pm), None)
                    evening_scan = next((s for s in reversed(scans) if s >= thirteen_pm), None)
                    if morning_scan:
                        morning_time = morning_scan.strftime('%H:%M:%S')
                    if evening_scan:
                        evening_time = evening_scan.strftime('%H:%M:%S')
                    if morning_scan and evening_scan:
                        status = "P"
                    elif morning_scan or evening_scan:
                        status = "HDP"
        full_history.append(
            {"date": date_str, "morning_status": morning_time, "evening_status": evening_time, "final_status": status})
        current_date -= timedelta(days=1)
    return jsonify(full_history)







@app.route('/api/get_member_stats/<path:team_name>/<enrollment_id>')
def api_get_member_stats(team_name, enrollment_id):
    """
    Calculates monthly and yearly attendance percentages for a member.
    This version now correctly calculates percentage based on actual working days
    (Days Analyzed - Total Leave Days).
    """
    if 'username' in session:
        leader_username = session['username']
    elif 'member_info' in session:
        leader_username = session['member_info']['leader_username']
    else:
        return jsonify({"error": "Unauthorized"}), 401

    now = datetime.now()

    # --- Calculate for the Current Month ---
    month_start = now.replace(day=1).date()
    month_end = now.date()
    month_analysis = compute_attendance_analysis(leader_username, team_name, enrollment_id, month_start, month_end)

    month_percent = 0.0
    if month_analysis.get("status") == "success":
        totals = month_analysis.get("totals", {})
        days_analyzed = month_analysis.get("days_analyzed", 0)

        # CORRECTED LOGIC: Calculate actual working days by subtracting leaves
        total_leave_days = totals.get("total_team_leaves", 0) + totals.get("total_approved_requests", 0)
        total_working_days = days_analyzed - total_leave_days

        if total_working_days > 0:
            present_score = (totals.get("total_present", 0) * 1.0) + (totals.get("total_half_day", 0) * 0.5)
            month_percent = (present_score / total_working_days) * 100

    # --- Calculate for the Current Year ---
    year_start = now.replace(month=1, day=1).date()
    year_end = now.date()
    year_analysis = compute_attendance_analysis(leader_username, team_name, enrollment_id, year_start, year_end)

    year_percent = 0.0
    if year_analysis.get("status") == "success":
        totals = year_analysis.get("totals", {})
        days_analyzed = year_analysis.get("days_analyzed", 0)

        # CORRECTED LOGIC: Calculate actual working days by subtracting leaves
        total_leave_days = totals.get("total_team_leaves", 0) + totals.get("total_approved_requests", 0)
        total_working_days = days_analyzed - total_leave_days

        if total_working_days > 0:
            present_score = (totals.get("total_present", 0) * 1.0) + (totals.get("total_half_day", 0) * 0.5)
            year_percent = (present_score / total_working_days) * 100

    return jsonify({
        "month_percent": f"{month_percent:.1f}%",
        "year_percent": f"{year_percent:.1f}%"
    })






@app.route('/api/mark_leave_for_all_teams', methods=['POST'])
@login_required
def api_mark_leave_for_all_teams():
    username = session['username']
    date_str = request.json.get('date')
    if not date_str:
        return jsonify({"status": "error", "message": "Date is required."})
    teams_path = os.path.join(get_user_path(username), 'teams')
    if not os.path.exists(teams_path):
        return jsonify({"status": "info", "message": "No teams found to mark leave for."})
    teams = [d for d in os.listdir(teams_path) if os.path.isdir(os.path.join(teams_path, d))]
    for team_name in teams:
        team_path = get_team_path(username, team_name)
        leave_file = os.path.join(team_path, 'leave_dates.json')
        leave_dates = json.load(open(leave_file)) if os.path.exists(leave_file) else []
        if date_str not in leave_dates:
            leave_dates.append(date_str)
            with open(leave_file, 'w') as f:
                json.dump(sorted(leave_dates), f, indent=4)
    # remove any cancellation notes for that date (since now it's globally marked again)
    canc = load_cancellations(username)
    new_canc = [c for c in canc if c.get('date') != date_str]
    if len(new_canc) != len(canc):
        save_cancellations(username, new_canc)
    return jsonify({"status": "success", "message": f"{date_str} has been marked as a leave day for all your teams."})

@app.route('/api/leader/get_member_history/<path:team_name>/<enrollment_id>')
@login_required
def leader_get_member_history(team_name, enrollment_id):
    leader_username = session['username']
    members_file = os.path.join(get_team_path(leader_username, team_name), 'team_members.json')
    if not os.path.exists(members_file):
        return jsonify([])
    with open(members_file, 'r') as f:
        members_data = json.load(f)
    member_info = members_data.get(enrollment_id)
    if not member_info:
        return jsonify([])
    logs_path = get_team_data_path(leader_username, team_name, 'attendance_logs')
    full_history = []
    if not os.path.exists(logs_path):
        return jsonify([])
    team_path = get_team_path(leader_username, team_name)
    leave_file = os.path.join(team_path, 'leave_dates.json')
    leave_dates = json.load(open(leave_file)) if os.path.exists(leave_file) else []


    # member-level leaves
    member_leaves = load_member_leaves(leader_username, team_name)



    enrollment_date = datetime.strptime(member_info['enrollment_date'], '%Y-%m-%d').date()
    end_date = datetime.now().date()
    all_logs = {}
    for log_filename in os.listdir(logs_path):
        if log_filename.endswith('.json'):
            date_str = log_filename.replace('.json', '')
            with open(os.path.join(logs_path, log_filename), 'r') as f:
                all_logs[date_str] = json.load(f)
    current_date = end_date
    while current_date >= enrollment_date:
        date_str = current_date.strftime('%Y-%m-%d')
        status, morning_time, evening_time = "A", "-", "-"

        # member-level override
        if date_str in member_leaves and enrollment_id in member_leaves[date_str]:
            status = "Leave"

        else:
            if date_str in leave_dates:
                status = "Leave"
            else:
                date_logs = all_logs.get(date_str)
                if date_logs and enrollment_id in date_logs:
                    thirteen_pm = dt_time(13, 0)
                    scans = sorted([datetime.strptime(t, '%H:%M:%S').time() for t in date_logs[enrollment_id]])
                    morning_scan = next((s for s in scans if s < thirteen_pm), None)
                    evening_scan = next((s for s in reversed(scans) if s >= thirteen_pm), None)
                    if morning_scan:
                        morning_time = morning_scan.strftime('%H:%M:%S')
                    if evening_scan:
                        evening_time = evening_scan.strftime('%H:%M:%S')
                    if morning_scan and evening_scan:
                        status = "P"
                    elif morning_scan or evening_scan:
                        status = "HDP"
        full_history.append(
            {"date": date_str, "morning_status": morning_time, "evening_status": evening_time, "final_status": status})
        current_date -= timedelta(days=1)
    return jsonify(full_history)






# --- New/Updated leave-request notification helpers & endpoints ---
 # add this near other imports at top if not already present

# Note: If you already have get_member_leaves_path/load_member_leaves/save_member_leaves/add_member_leave_exception
# from earlier patches, keep them. They are used here to manage member-level approved leaves.

def _read_leave_requests_for_team(username: str, team_name: str):
    """
    Return list of request dicts from team_path/leave_requests.json
    Always returns a list (possibly empty).
    """
    requests_file = os.path.join(get_team_path(username, team_name), 'leave_requests.json')
    if not os.path.exists(requests_file):
        return []
    try:
        with open(requests_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"Warning: failed to read leave_requests.json for {username}/{team_name}: {e}")
        return []

def _write_leave_requests_for_team(username: str, team_name: str, requests_list: list):
    requests_file = os.path.join(get_team_path(username, team_name), 'leave_requests.json')
    tmp = requests_file + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(requests_list, f, indent=4)
        os.replace(tmp, requests_file)
        return True
    except Exception as e:
        print(f"Warning: failed to write leave_requests.json for {username}/{team_name}: {e}")
        return False

# -------------------------
# Member: submit leave req.
# -------------------------
@app.route('/api/member/request_leave', methods=['POST'])
@member_login_required
def api_member_request_leave():
    """
    Member sends leave request.
    Payload JSON: { date: 'YYYY-MM-DD', note: '...' }
    Stores request object under team folder leave_requests.json
    """
    member = session.get('member_info')
    if not member:
        return jsonify({"status": "error", "message": "Member not authenticated."}), 401

    data = request.get_json() or {}
    date_str = data.get('date')
    note = (data.get('note') or '').strip()

    if not date_str:
        return jsonify({"status": "error", "message": "Date is required."}), 400

    leader_username = member['leader_username']
    team_name = member['team_name']
    os.makedirs(get_team_path(leader_username, team_name), exist_ok=True)

    requests_list = _read_leave_requests_for_team(leader_username, team_name)

    req_obj = {
        "id": str(uuid.uuid4()),
        "team_name": team_name,
        "leader_username": leader_username,
        "enrollment_id": member['enrollment_id'],
        "member_name": member.get('name'),
        "date": date_str,
        "note": note,
        "status": "pending",               # pending/accepted/rejected
        "created_at": datetime.now().isoformat(),
        "responded_at": None,
        "reply_note": None,
        "responded_by": None,
        # Visibility & member-side flags:
        "visible_to_member": False,        # only show to member after leader replies
        "member_seen": False               # whether member has acknowledged the leader reply
    }
    requests_list.append(req_obj)
    _write_leave_requests_for_team(leader_username, team_name, requests_list)
    return jsonify({"status": "success", "message": "Leave request submitted.", "request": req_obj})

# -------------------------
# Leader: fetch requests
# -------------------------
@app.route('/api/get_leave_requests')
@login_required
def api_get_leave_requests():
    """
    Leader fetches leave requests. Supports optional query params:
      - team_name
      - enrollment_id
      - status (pending|accepted|rejected)
    Returns matching request objects across the leader's teams.
    """
    username = session['username']
    team_filter = request.args.get('team_name')
    enrollment_filter = request.args.get('enrollment_id')
    status_filter = request.args.get('status')  # optional: pending/accepted/rejected

    teams_root = os.path.join(get_user_path(username), 'teams')
    if not os.path.exists(teams_root):
        return jsonify([])

    results = []
    for t in os.listdir(teams_root):
        if team_filter and t != team_filter:
            continue
        requests_list = _read_leave_requests_for_team(username, t)
        for r in requests_list:
            if enrollment_filter and r.get('enrollment_id') != enrollment_filter:
                continue
            if status_filter and r.get('status') != status_filter:
                continue
            results.append(r)
    # most recent first
    results.sort(key=lambda x: x.get('created_at') or '', reverse=True)
    return jsonify(results)

# -------------------------
# Leader: accept/reject request (updated)
# -------------------------
@app.route('/api/respond_leave_request', methods=['POST'])
@login_required
def api_respond_leave_request():
    """
    Leader responds to request.
    Payload: { id: <request_id>, action: 'accept'|'reject', reply_note: '...' }
    If accepted -> add_member_leave_exception is called to create member-level leave.
    Also sets visible_to_member=True & member_seen=False so the member receives a notification.
    """
    username = session['username']
    data = request.get_json() or {}
    request_id = data.get('id')
    action = data.get('action')
    reply_note = (data.get('reply_note') or '').strip()

    if not request_id or action not in ('accept', 'reject'):
        return jsonify({"status": "error", "message": "Invalid payload."}), 400

    teams_root = os.path.join(get_user_path(username), 'teams')
    if not os.path.exists(teams_root):
        return jsonify({"status": "error", "message": "No teams found."}), 404

    # Find the request across teams
    for t in os.listdir(teams_root):
        requests_list = _read_leave_requests_for_team(username, t)
        updated = False
        for r in requests_list:
            if r.get('id') == request_id:
                old_status = r.get('status')
                r['status'] = 'accepted' if action == 'accept' else 'rejected'
                r['responded_at'] = datetime.now().isoformat()
                r['reply_note'] = reply_note
                r['responded_by'] = username
                r['visible_to_member'] = True
                r['member_seen'] = False

                # If accepted -> add the member-level leave exception so attendance functions honor it
                if action == 'accept':
                    try:
                        add_member_leave_exception(username, t, r.get('date'), r.get('enrollment_id'))
                    except Exception as e:
                        print("Warning: add_member_leave_exception failed:", e)
                else:
                    # if rejecting, ensure any previous exception is removed (safety)
                    try:
                        remove_member_leave_exception(username, t, r.get('date'), r.get('enrollment_id'))
                    except Exception as e:
                        print("Warning: remove_member_leave_exception failed:", e)

                updated = True
                break
        if updated:
            _write_leave_requests_for_team(username, t, requests_list)
            return jsonify({"status": "success", "message": f"Request {action}ed.", "request": r})
    return jsonify({"status": "error", "message": "Request not found."}), 404

# -------------------------
# Member: get notifications (leader replies)
# -------------------------
@app.route('/api/get_member_notifications')
@member_login_required
def api_get_member_notifications():
    """
    Returns requests visible to this member with status accepted/rejected (leader responded).
    Member sees only requests with visible_to_member == True.
    """
    member = session.get('member_info')
    if not member:
        return jsonify([])

    leader_username = member['leader_username']
    team_name = member['team_name']
    enrollment_id = member['enrollment_id']

    requests_list = _read_leave_requests_for_team(leader_username, team_name)
    results = [r for r in requests_list if r.get('enrollment_id') == enrollment_id and r.get('visible_to_member', False) and r.get('status') in ('accepted', 'rejected')]
    # order by responded_at/or created_at
    results.sort(key=lambda x: x.get('responded_at') or x.get('created_at') or '', reverse=True)
    return jsonify(results)

# -------------------------
# Member: mark notification as read (optional)
# -------------------------
@app.route('/api/acknowledge_member_notification', methods=['POST'])
@member_login_required
def api_acknowledge_member_notification():
    """
    Mark a notification as 'seen' by member.
    Payload: { id: <request_id> }
    """
    member = session.get('member_info')
    if not member:
        return jsonify({"status": "error", "message": "Member not authenticated."}), 401
    data = request.get_json() or {}
    req_id = data.get('id')
    if not req_id:
        return jsonify({"status": "error", "message": "Request id required."}), 400

    leader_username = member['leader_username']
    team_name = member['team_name']
    enrollment_id = member['enrollment_id']

    requests_list = _read_leave_requests_for_team(leader_username, team_name)
    updated = False
    for r in requests_list:
        if r.get('id') == req_id and r.get('enrollment_id') == enrollment_id:
            r['member_seen'] = True
            updated = True
            break
    if updated:
        _write_leave_requests_for_team(leader_username, team_name, requests_list)
        return jsonify({"status": "success", "message": "Marked as read."})
    return jsonify({"status": "error", "message": "Notification not found."}), 404

# -------------------------
# Member: delete/hide notification
# -------------------------
@app.route('/api/delete_member_notification', methods=['POST'])
@member_login_required
def api_delete_member_notification():
    """
    Hide a notification from member dashboard (does not delete leader's record).
    Payload: { id: <request_id> }
    """
    member = session.get('member_info')
    if not member:
        return jsonify({"status": "error", "message": "Member not authenticated."}), 401

    data = request.get_json() or {}
    req_id = data.get('id')
    if not req_id:
        return jsonify({"status": "error", "message": "Request id required."}), 400

    leader_username = member['leader_username']
    team_name = member['team_name']
    enrollment_id = member['enrollment_id']

    requests_list = _read_leave_requests_for_team(leader_username, team_name)
    updated = False
    for r in requests_list:
        if r.get('id') == req_id and r.get('enrollment_id') == enrollment_id:
            # Prefer to keep record but hide from member's list
            r['visible_to_member'] = False
            updated = True
            break
    if updated:
        _write_leave_requests_for_team(leader_username, team_name, requests_list)
        return jsonify({"status": "success", "message": "Notification hidden."})
    return jsonify({"status": "error", "message": "Notification not found."}), 404

# -------------------------
# Leader: quick endpoint to get pending count for UI badge
# (you might already have /api/get_pending_leave_count; if so keep that - it's equivalent)
# -------------------------
@app.route('/api/get_pending_leave_count')
@login_required
def api_get_pending_leave_count():
    username = session['username']
    teams_root = os.path.join(get_user_path(username), 'teams')
    count = 0
    if os.path.exists(teams_root):
        for t in os.listdir(teams_root):
            requests_list = _read_leave_requests_for_team(username, t)
            for r in requests_list:
                if r.get('status') == 'pending':
                    count += 1
    return jsonify({"pending": count})




# --------------------------
# Member Leave Request APIs
# --------------------------
def _get_leave_requests_path(username):
    """Return path to leader's leave requests file."""
    return os.path.join(get_user_path(username), 'leave_requests.json')

def _read_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default
    except Exception:
        return default

def _write_json_safe(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


@app.route('/api/get_member_leave_exceptions/<team_name>')
@login_required
def api_get_member_leave_exceptions(team_name):
    """
    Return member-specific accepted leave mapping for given team.
    Response: { "YYYY-MM-DD": ["enroll1","enroll2"], ... }
    """
    username = session['username']
    team_path = get_team_path(username, team_name)
    member_leaves_file = os.path.join(team_path, 'member_leaves.json')
    member_leaves = _read_json_safe(member_leaves_file, {})
    return jsonify(member_leaves)




# ---------------------------
# NEW: Leave cycles and calendar events (added)
# ---------------------------

def generate_dates_for_cycle(cycle):
    """
    Generate concrete date list (YYYY-MM-DD) for a cycle.
    Supported cycle types:
      - manual: every date from start_date to end_date (inclusive)
      - weekly: weekly recurring; weekly_mode 'by-weekday' or 'by-monthday'
         - by-weekday: cycle['days'] = ["Mon","Tue",...]
         - by-monthday: cycle['month_days'] = ["1","15","30"] repeat on those days each month between start/end
    """
    start = datetime.strptime(cycle['start_date'], '%Y-%m-%d').date()
    end = datetime.strptime(cycle['end_date'], '%Y-%m-%d').date()
    res = []
    if cycle['type'] == 'manual':
        cur = start
        while cur <= end:
            res.append(cur.strftime('%Y-%m-%d'))
            cur += timedelta(days=1)
        return sorted(list(set(res)))
    if cycle['type'] == 'weekly':
        mode = cycle.get('weekly_mode', 'by-weekday')
        if mode == 'by-weekday':
            mapping = {'Mon':0,'Tue':1,'Wed':2,'Thu':3,'Fri':4,'Sat':5,'Sun':6}
            selected = set(mapping[d] for d in (cycle.get('days') or []))
            cur = start
            while cur <= end:
                if cur.weekday() in selected:
                    res.append(cur.strftime('%Y-%m-%d'))
                cur += timedelta(days=1)
            return sorted(list(set(res)))
        elif mode == 'by-monthday':
            mdays = [int(x) for x in (cycle.get('month_days') or []) if x and x.isdigit()]
            y = start.year; m = start.month
            while (y < end.year) or (y == end.year and m <= end.month):
                for md in mdays:
                    try:
                        dt = date(y, m, md)
                    except Exception:
                        continue
                    if start <= dt <= end:
                        res.append(dt.strftime('%Y-%m-%d'))
                m += 1
                if m > 12:
                    m = 1; y += 1
            return sorted(list(set(res)))
    return []




@app.route('/api/get_leave_cycles')
@login_required
def api_get_leave_cycles():
    username = session['username']
    cycles = load_cycles(username)
    return jsonify(cycles)

@app.route('/api/create_leave_cycle', methods=['POST'])
@login_required
def api_create_leave_cycle():
    username = session['username']
    payload = request.get_json() or {}
    # validate required fields
    payload['id'] = str(uuid.uuid4())
    payload['start_date'] = norm_ymd(payload.get('start_date'))
    payload['end_date'] = norm_ymd(payload.get('end_date'))
    if not payload.get('start_date') or not payload.get('end_date') or not payload.get('type'):
        return jsonify({"status":"error","message":"start_date, end_date and type are required."}), 400
    cycles = load_cycles(username)
    cycles.append(payload)
    save_cycles(username, cycles)
    # expand and write dates into team files
    dates = generate_dates_for_cycle(payload)
    if payload.get('scope') == 'global':
        teams_dir = os.path.join(get_user_path(username), 'teams')
        if os.path.exists(teams_dir):
            for t in os.listdir(teams_dir):
                f = os.path.join(get_team_path(username, t), 'leave_dates.json')
                arr = json.load(open(f)) if os.path.exists(f) else []
                arr_set = set(arr)
                for d in dates:
                    arr_set.add(d)
                with open(f, 'w') as fh:
                    json.dump(sorted(list(arr_set)), fh, indent=4)
    else:
        team = payload.get('team_name')
        if not team:
            return jsonify({"status":"error","message":"team_name is required for non-global cycles."}),400
        f = os.path.join(get_team_path(username, team), 'leave_dates.json')
        arr = json.load(open(f)) if os.path.exists(f) else []
        arr_set = set(arr)
        for d in dates:
            arr_set.add(d)
        with open(f, 'w') as fh:
            json.dump(sorted(list(arr_set)), fh, indent=4)
    return jsonify({"status":"success","message":"Cycle created.","id":payload['id']})



@app.route('/api/edit_leave_cycle', methods=['POST'])
@login_required
def api_edit_leave_cycle():
    username = session['username']
    payload = request.get_json() or {}
    cid = payload.get('id')
    if not cid:
        return jsonify({"status":"error","message":"id required"}), 400
    cycles = load_cycles(username)
    orig = next((c for c in cycles if c.get('id') == cid), None)
    if not orig:
        return jsonify({"status":"error","message":"Cycle not found"}), 404
    # remove orig generated dates from teams
    orig_dates = generate_dates_for_cycle(orig)
    if orig.get('scope') == 'global':
        teams_dir = os.path.join(get_user_path(username), 'teams')
        if os.path.exists(teams_dir):
            for t in os.listdir(teams_dir):
                f = os.path.join(get_team_path(username, t), 'leave_dates.json')
                arr = json.load(open(f)) if os.path.exists(f) else []
                arr_set = set(arr)
                for d in orig_dates:
                    arr_set.discard(d)
                with open(f, 'w') as fh:
                    json.dump(sorted(list(arr_set)), fh, indent=4)
    else:
        team = orig.get('team_name')
        if team:
            f = os.path.join(get_team_path(username, team), 'leave_dates.json')
            arr = json.load(open(f)) if os.path.exists(f) else []
            arr_set = set(arr)
            for d in orig_dates:
                arr_set.discard(d)
            with open(f, 'w') as fh:
                json.dump(sorted(list(arr_set)), fh, indent=4)
    # update orig metadata
    for k in ['scope','team_name','type','weekly_mode','days','month_days','start_date','end_date']:
        if k in payload:
            orig[k] = payload[k]
    save_cycles(username, cycles)
    # add new generated dates
    new_dates = generate_dates_for_cycle(orig)
    if orig.get('scope') == 'global':
        teams_dir = os.path.join(get_user_path(username), 'teams')
        if os.path.exists(teams_dir):
            for t in os.listdir(teams_dir):
                f = os.path.join(get_team_path(username, t), 'leave_dates.json')
                arr = json.load(open(f)) if os.path.exists(f) else []
                arr_set = set(arr)
                for d in new_dates:
                    arr_set.add(d)
                with open(f, 'w') as fh:
                    json.dump(sorted(list(arr_set)), fh, indent=4)
    else:
        team = orig.get('team_name')
        if team:
            f = os.path.join(get_team_path(username, team), 'leave_dates.json')
            arr = json.load(open(f)) if os.path.exists(f) else []
            arr_set = set(arr)
            for d in new_dates:
                arr_set.add(d)
            with open(f, 'w') as fh:
                json.dump(sorted(list(arr_set)), fh, indent=4)
    return jsonify({"status":"success","message":"Cycle edited."})

@app.route('/api/delete_leave_cycle', methods=['POST'])
@login_required
def api_delete_leave_cycle():
    username = session['username']
    payload = request.get_json() or {}
    cid = payload.get('id')
    cycles = load_cycles(username)
    idx = next((i for i,c in enumerate(cycles) if c.get('id')==cid), None)
    if idx is None:
        return jsonify({"status":"error","message":"Not found"}), 404
    cycle = cycles.pop(idx)
    # remove generated dates
    dates = generate_dates_for_cycle(cycle)
    if cycle.get('scope') == 'global':
        teams_dir = os.path.join(get_user_path(username), 'teams')
        if os.path.exists(teams_dir):
            for t in os.listdir(teams_dir):
                f = os.path.join(get_team_path(username, t), 'leave_dates.json')
                arr = json.load(open(f)) if os.path.exists(f) else []
                arr_set = set(arr)
                for d in dates:
                    if d in arr_set:
                        arr_set.discard(d)
                with open(f, 'w') as fh:
                    json.dump(sorted(list(arr_set)), fh, indent=4)
    else:
        team = cycle.get('team_name')
        if team:
            f = os.path.join(get_team_path(username, team), 'leave_dates.json')
            arr = json.load(open(f)) if os.path.exists(f) else []
            arr_set = set(arr)
            for d in dates:
                if d in arr_set:
                    arr_set.discard(d)
            with open(f, 'w') as fh:
                json.dump(sorted(list(arr_set)), fh, indent=4)
    save_cycles(username, cycles)
    # Also remove cancellations for those dates (as cycle deleted -> fully remove)
    canc = load_cancellations(username)
    new_canc = [c for c in canc if c.get('date') not in dates]
    if len(new_canc) != len(canc):
        save_cancellations(username, new_canc)
    return jsonify({"status":"success","message":"Cycle deleted."})

@app.route('/api/get_calendar_events')
@login_required
def api_get_calendar_events():
    """
    Returns calendar events for the requested year/month.
    Response: [{date: "YYYY-MM-DD", items: [{team, title}], title: "teams joined by comma"}, ...]
    Also includes cancellation notes (team-specific) in separate 'cancellations' file; front-end can merge/annotate.
    """
    username = session['username']
    try:
        year = int(request.args.get('year', datetime.now().year))
        month = int(request.args.get('month', datetime.now().month))
    except Exception:
        year = datetime.now().year; month = datetime.now().month
    teams_dir = os.path.join(get_user_path(username), 'teams')
    events = []
    if os.path.exists(teams_dir):
        for t in os.listdir(teams_dir):
            f = os.path.join(get_team_path(username, t), 'leave_dates.json')
            arr = json.load(open(f)) if os.path.exists(f) else []
            for d in arr:
                try:
                    dt = datetime.strptime(d, '%Y-%m-%d')
                except Exception:
                    continue
                if dt.year == year and dt.month == month:
                    events.append({"date": d, "team": t, "title": "Leave"})
    # consolidate by date
    by_date = {}
    for e in events:
        by_date.setdefault(e['date'], []).append({"team": e['team'], "title": e['title']})
    out = []
    for d, items in by_date.items():
        out.append({"date": d, "items": items, "title": ", ".join([i.get('team') or '' for i in items])})
    # Also include cancellations (team-specific notes)
    canc = load_cancellations(username)
    # Only include cancellations relevant to the requested month/year
    cancell_for_month = [c for c in canc if c.get('date')[:7] == f"{year:04d}-{month:02d}"]
    # attach cancellations: the frontend can request /api/get_calendar_events and /api/get_leave_cancellations
    return jsonify(out)

@app.route('/api/get_leave_cancellations')
@login_required
def api_get_leave_cancellations():
    username = session['username']
    return jsonify(load_cancellations(username))

# ---------------------------
# End of new features
# ---------------------------

# all previously present API endpoints remain intact after this point.



# --- Attendance Analysis helpers & endpoints ---
# Paste below any imports at top if missing:
# from datetime import datetime, timedelta, time as dt_time
# (these are already in your file per your provided app.py)

def _parse_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except Exception:
        return None

def _is_weekend(d):
    # 5 = Saturday, 6 = Sunday
    return d.weekday() >= 5

def load_member_leaves(username: str, team_name: str):
    """
    Return a dict mapping date_str -> list of enrollment_ids that have member-level approved leaves.
    File expected at teams/<team>/member_leaves.json
    If file missing, returns {}.
    """
    member_leaves_file = os.path.join(get_team_path(username, team_name), 'member_leaves.json')
    try:
        if os.path.exists(member_leaves_file):
            data = json.load(open(member_leaves_file))
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"Warning: could not read member_leaves.json: {e}")
    return {}

def read_leave_requests_for_team(username: str, team_name: str):
    """
    Return list of leave request objects stored at teams/<team>/leave_requests.json
    If missing or malformed, returns [].
    """
    req_file = os.path.join(get_team_path(username, team_name), 'leave_requests.json')
    try:
        if os.path.exists(req_file):
            data = json.load(open(req_file))
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"Warning: failed to read leave_requests.json: {e}")
    return []

# ---------- REPLACE compute_attendance_analysis + endpoints WITH THIS BLOCK ----------
def compute_attendance_analysis(username: str, team_name: str, enrollment_id: str, start_date, end_date):
    """
    Core computation. This version now uses the full attendance criteria rules
    to ensure 100% consistency with the daily final report.
    """
    members_file = os.path.join(get_team_path(username, team_name), 'team_members.json')
    if not os.path.exists(members_file): return {"error": "Team not found"}
    members_data = json.load(open(members_file))
    if enrollment_id not in members_data: return {"error": "Member not found"}

    member_info = members_data[enrollment_id]
    enrollment_date = _parse_date(member_info.get('enrollment_date'))
    leaving_date = _parse_date(member_info.get('leaving_date'))
    today = datetime.now().date()
    analysis_upper_bound = min(end_date, today)
    effective_start = max(start_date, enrollment_date) if enrollment_date else start_date

    if analysis_upper_bound < effective_start:
        return {"status": "success", "days_analyzed": 0, "totals": {}}

    # --- Load All Necessary Data and Rules ---
    rules_file = os.path.join(get_team_path(username, team_name), 'attendance_rules.json')
    rules = read_json_file(rules_file, get_default_attendance_rules())
    team_leaves = read_json_file(os.path.join(get_team_path(username, team_name), 'leave_dates.json'), [])
    member_leaves = load_member_leaves(username, team_name)
    requests_list = read_leave_requests_for_team(username, team_name)
    logs_path = get_team_data_path(username, team_name, 'attendance_logs')

    totals = {
        "total_present": 0, "total_absent": 0, "total_late_coming": 0, "total_early_going": 0,
        "total_half_day": 0, "total_team_leaves": 0, "total_requests": 0,
        "total_approved_requests": 0, "total_rejected_requests": 0, "total_inactive_days": 0,
        "total_working_seconds": 0
    }
    days_analyzed = 0
    cur = effective_start
    while cur <= analysis_upper_bound:
        date_str = cur.strftime('%Y-%m-%d')
        days_analyzed += 1

        if (leaving_date and cur > leaving_date) or (enrollment_date and cur < enrollment_date):
            totals['total_inactive_days'] += 1
        elif date_str in team_leaves:
            totals['total_team_leaves'] += 1
        elif date_str in member_leaves and enrollment_id in member_leaves.get(date_str, []):
            pass  # This day is an approved personal leave, counted by the request counter later
        else:
            log_file = os.path.join(logs_path, f"{date_str}.json")
            date_logs = read_json_file(log_file, {})
            member_scans = date_logs.get(enrollment_id, [])

            if len(member_scans) < 2:
                totals['total_absent'] += 1
            else:
                # --- START: Full Rules-Based Calculation Logic ---
                parsed = sorted([datetime.strptime(t, '%H:%M:%S') for t in member_scans])
                time_in_dt, time_out_dt = parsed[0], parsed[-1]
                total_seconds = (time_out_dt - time_in_dt).total_seconds()
                totals['total_working_seconds'] += total_seconds

                status, is_late, is_early = 'Absent', False, False
                criteria_type = rules.get('criteria_type', 'time')

                if criteria_type == 'hours':
                    work_hours_float = total_seconds / 3600.0
                    min_full = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_present', 8.0))
                    min_early = float(rules.get('full_day', {}).get('by_hours', {}).get('min_hours_early_go', 7.0))
                    min_half = float(rules.get('half_day', {}).get('by_hours', {}).get('min_hours', 4.0))

                    if work_hours_float >= min_full:
                        status = 'Present'
                    elif work_hours_float >= min_early:
                        status = 'Present'; is_early = True
                    elif work_hours_float >= min_half:
                        status = 'Half Day'
                else:  # Time-based
                    def to_time(t_str):
                        try:
                            return dt_time.fromisoformat(t_str) if t_str else None
                        except:
                            return None

                    safe_end = to_time(rules.get('half_day', {}).get('by_time', {}).get('in_time_safe_range_end'))
                    hd_out = to_time(rules.get('half_day', {}).get('by_time', {}).get('required_out_time'))
                    fd_early = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_early_go'))
                    fd_full = to_time(rules.get('full_day', {}).get('by_time', {}).get('required_out_time_present'))

                    if safe_end and time_in_dt.time() > safe_end: is_late = True
                    if fd_early and time_out_dt.time() < fd_early: is_early = True

                    if fd_full and time_out_dt.time() >= fd_full:
                        status = 'Present'
                    elif fd_early and time_out_dt.time() >= fd_early:
                        status = 'Present'
                    elif hd_out and time_out_dt.time() >= hd_out:
                        status = 'Half Day'

                # Increment counters based on final status
                if status == 'Present':
                    totals['total_present'] += 1
                elif status == 'Half Day':
                    totals['total_half_day'] += 1
                else:
                    totals['total_absent'] += 1

                if status != 'Absent':
                    if is_late: totals['total_late_coming'] += 1
                    if is_early: totals['total_early_going'] += 1
                # --- END: Full Rules-Based Calculation Logic ---

        cur += timedelta(days=1)

    # Count leave requests
    for req in requests_list:
        req_date = _parse_date(req.get('date'))
        if req.get('enrollment_id') == enrollment_id and req_date and start_date <= req_date <= end_date:
            totals['total_requests'] += 1
            if req.get('status') == 'accepted':
                totals['total_approved_requests'] += 1
            elif req.get('status') == 'rejected':
                totals['total_rejected_requests'] += 1

    # Format total working hours
    total_h = totals['total_working_seconds'] // 3600
    total_m = (totals['total_working_seconds'] % 3600) // 60
    totals['total_working_hours'] = f"{int(total_h)}h {int(total_m)}m"
    del totals['total_working_seconds']

    return {"status": "success", "days_analyzed": days_analyzed, "totals": totals}


# Leader endpoint (analyze member)
@app.route('/api/attendance_analysis/<team_name>/<enrollment_id>')
@login_required
def api_attendance_analysis(team_name, enrollment_id):
    username = session['username']
    start_s = request.args.get('start')
    end_s = request.args.get('end')
    start_date = _parse_date(start_s)
    end_date = _parse_date(end_s)
    if not start_date or not end_date or start_date > end_date:
        return jsonify({"status": "error", "message": "Provide valid start and end dates (YYYY-MM-DD)."}), 400

    result = compute_attendance_analysis(username, team_name, enrollment_id, start_date, end_date)
    if 'error' in result:
        return jsonify({"status": "error", "message": result['error']}), 404
    return jsonify(result)


# Member endpoint (self analysis)
@app.route('/api/attendance_analysis_self')
@member_login_required
def api_attendance_analysis_self():
    member = session.get('member_info')
    if not member:
        return jsonify({"status": "error", "message": "Unauthorized."}), 401
    leader_username = member.get('leader_username')
    team_name = member.get('team_name')
    enrollment_id = member.get('enrollment_id')
    if not all([leader_username, team_name, enrollment_id]):
        return jsonify({"status":"error","message":"Member context incomplete."}), 500

    start_s = request.args.get('start')
    end_s = request.args.get('end')
    start_date = _parse_date(start_s)
    end_date = _parse_date(end_s)
    if not start_date or not end_date or start_date > end_date:
        return jsonify({"status": "error", "message": "Provide valid start and end dates (YYYY-MM-DD)."}), 400

    result = compute_attendance_analysis(leader_username, team_name, enrollment_id, start_date, end_date)
    if 'error' in result:
        return jsonify({"status": "error", "message": result['error']}), 404
    return jsonify(result)
# ---------- END REPLACEMENT BLOCK ----------


# ---------------------------
# Run server
# ---------------------------
if __name__ == '__main__':
    # Enables debugging features like auto-reloading and detailed error pages.
    # Should be set to False in a production environment.
    app.run(debug=True)
