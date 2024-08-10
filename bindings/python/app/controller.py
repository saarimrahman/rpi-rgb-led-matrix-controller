from moviepy.editor import  ColorClip, TextClip, CompositeVideoClip
import os
import cv2
import numpy as np
import threading
from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import json
from colorsys import rgb_to_hsv

# Configuration for the matrix
options = RGBMatrixOptions()
options.cols = 64
options.rows = 32
options.chain_length = 2
options.parallel = 3
options.brightness = 75
options.pwm_bits = 11
options.gpio_slowdown = 4.0
options.show_refresh_rate = 0
options.hardware_mapping = 'regular'
options.pwm_dither_bits = 2
options.drop_privileges = False

matrix = RGBMatrix(options=options)

canvas_w = 2 * options.cols
canvas_h = 3 * options.rows

# Global variables
filename = ''
stop_video = True
queue = []
start_time = float('-inf')
current_speed = 1.0
loop_queue = False

QUEUE_PLAY_TIME = 30
PLAYLIST_FOLDER = 'playlists'
UPLOAD_FOLDER = 'media'
LOGS_FOLDER = 'logs'
VIDEO_EXTENSIONS = {'mp4', 'avi', 'gif', 'mov'}
IMG_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMG_EXTENSIONS

os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(PLAYLIST_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Predefined list of common colors and their hex codes
COMMON_COLORS = {
  'black': '#000000',
  'white': '#FFFFFF',
  'red': '#FF0000',
  'green': '#008000',
  'blue': '#0000FF',
  'yellow': '#FFFF00',
  'cyan': '#00FFFF',
  'magenta': '#FF00FF',
  'gray': '#808080',
  'maroon': '#800000',
  'olive': '#808000',
  'purple': '#800080',
  'teal': '#008080',
  'navy': '#000080',
  'orange': '#FFA500',
  'pink': '#FFC0CB',
  'brown': '#A52A2A',
  'coral': '#FF7F50',
  'indigo': '#4B0082',
  'turquoise': '#40E0D0',
  'violet': '#EE82EE',
  'gold': '#FFD700',
  'silver': '#C0C0C0',
  'lime': '#00FF00',
  'crimson': '#DC143C',
  'khaki': '#F0E68C',
  'plum': '#DDA0DD',
  'ivory': '#FFFFF0',
  'lavender': '#E6E6FA'
}


app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLAYLIST_FOLDER'] = PLAYLIST_FOLDER


def play_video(cap, double_buffer, canvas_w, canvas_h):
    global stop_video, current_speed
    # Get the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set a default FPS if the video doesn't have a valid one
    if fps <= 0 or math.isnan(fps):
        default_fps = 30.0  # You can adjust this value as needed
        print(f'Warning: Invalid FPS detected. Using default FPS of {default_fps}')
        fps = default_fps

    frame_time = 1 / (fps * current_speed)

    while cap.isOpened() and not stop_video and not is_queue_media_expired():
        start = time.time()

        # Process first frame
        ret, im = cap.read()
        if not ret or stop_video:
            return
        im = cv2.resize(im, (canvas_w, canvas_h))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_pil1 = Image.fromarray(im)

        # Process second frame
        ret, im = cap.read()
        if not ret or stop_video or is_queue_media_expired():
            return
        im = cv2.resize(im, (canvas_w, canvas_h))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_pil2 = Image.fromarray(im)

        # Set images and swap buffer
        double_buffer.SetImage(im_pil1)
        double_buffer.SetImage(im_pil2, canvas_w)
        double_buffer = matrix.SwapOnVSync(double_buffer)

        # Calculate the time elapsed for processing and displaying frames
        elapsed = time.time() - start

        # Sleep for the remaining time to maintain the correct frame rate
        # Multiply frame_time by 2 since we're processing two frames
        sleep_time = max(0, (frame_time * 2) - elapsed)
        time.sleep(sleep_time)


def display_image(image, double_buffer, canvas_w, canvas_h):
    global stop_video
    if stop_video or is_queue_media_expired():
        return
    image = cv2.resize(image, (canvas_w, canvas_h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pils = []
    image_pil = Image.fromarray(image)
    for _ in range(2):
        img_pils.append(image_pil)

    double_buffer.SetImage(img_pils[0])
    double_buffer.SetImage(img_pils[1], canvas_w)
    double_buffer = matrix.SwapOnVSync(double_buffer)

    # Keep the image displayed until stop_video is True
    while not stop_video and not is_queue_media_expired():
        time.sleep(0.01)  # Small sleep to prevent busy-waiting


def is_queue_media_expired():
    global stop_video, queue, start_time
    if queue and time.time() - start_time > QUEUE_PLAY_TIME:
        toggle_stop_video()
        return True
    return False


def media_loop():
    global filename, stop_video, queue, start_time, loop_queue
    while True:
        while not stop_video:
            if queue and is_queue_media_expired() or not filename:
                filename = queue.pop(0)
                if loop_queue:
                    queue.append(filename)
                start_time = time.time()
            if not filename:
                print('Waiting for file...')
                time.sleep(1)
                break
            _, file_extension = os.path.splitext(filename)
            file_extension = file_extension.lstrip('.').lower()
            print(f'Playing {filename}' + (' from queue' if queue else ''))
            media_filepath = os.path.join('media', filename)
            double_buffer = matrix.CreateFrameCanvas()

            if file_extension in VIDEO_EXTENSIONS or file_extension == 'webp' and is_webp_video(media_filepath):
                cap = cv2.VideoCapture(media_filepath)
                play_video(cap, double_buffer, canvas_w, canvas_h)
            elif file_extension in IMG_EXTENSIONS:
                image = cv2.imread(media_filepath)
                display_image(image, double_buffer, canvas_w, canvas_h)
            else:
                raise ValueError(f'Unsupported file extension: {file_extension}')
            time.sleep(0.1)

        if filename == 'exit':
            break
        time.sleep(0.1)


def is_webp_video(file_path):
  with open(file_path, 'rb') as f:
      # Read the first 12 bytes
      header = f.read(12)

      # Check if it's a WebP file
      if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
          # Read the next 4 bytes
          chunk_header = f.read(4)

          # If it's 'VP8X', it might be an animated WebP
          if chunk_header == b'VP8X':
              flags = ord(f.read(1))
              # Check if the animation bit is set
              return bool(flags & 0b00000010)

          # If it's 'ANIM', it's definitely an animated WebP
          elif chunk_header == b'ANIM':
              return True

  # If we reach here, it's either a static WebP image or not a WebP file at all
  return False


def create_scrolling_text_video(text: str, filename: str, bg_color: tuple, text_color: str, font_size=150):
    filename = secure_filename(filename)
    output_path = os.path.join(UPLOAD_FOLDER, filename)

    base_duration = 5  # Duration of the video in seconds
    additional_duration_per_char = 1
    duration = base_duration + len(text) * additional_duration_per_char
    resolution = (1280, 720)  # Resolution of the video (width, height)
    background_clip = ColorClip(resolution, color=bg_color, duration=duration)

    # Create a TextClip with the loaded text
    text_clip = (TextClip(text, fontsize=font_size, color=text_color).set_duration(duration))

    # Calculate the x-coordinate to start off-screen on the right
    x_start = background_clip.size[0] + text_clip.size[0]

    # Set the position off-screen on the right
    text_clip = text_clip.set_position((x_start, 'center'))

    # Animate text from right to left
    def scroll_text_horizontal(t):
        x_position = x_start - (background_clip.size[0] + text_clip.size[0]) * 2 * t / duration
        return x_position, 'center'

    # Apply the scrolling animation
    text_clip = text_clip.set_position(scroll_text_horizontal)

    # Overlay text on the background clip
    result = CompositeVideoClip([background_clip, text_clip])
    result.write_videofile(output_path, fps=24)
    return filename


def allowed_file(filename):
  _, file_extension = os.path.splitext(filename)
  return '.' in filename and file_extension.lower().lstrip('.') in ALLOWED_EXTENSIONS


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def color_difference(color1, color2):
  # Convert RGB to HSV
  hsv1 = rgb_to_hsv(*(c/255 for c in color1))
  hsv2 = rgb_to_hsv(*(c/255 for c in color2))

  # Calculate differences
  h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])) * 2
  s_diff = abs(hsv1[1] - hsv2[1])
  v_diff = abs(hsv1[2] - hsv2[2])

  # Weighted sum of differences
  return math.sqrt(h_diff**2 * 4 + s_diff**2 * 1 + v_diff**2 * 2)


def approximate_color_name(hex_code):
  rgb_color = hex_to_rgb(hex_code)
  closest_color = min(COMMON_COLORS.keys(), key=lambda color: color_difference(rgb_color, hex_to_rgb(COMMON_COLORS[color])))
  return closest_color


@app.route('/')
def index():
    media_files = sorted(os.listdir('media'))
    return render_template('index.html', media_files=media_files)


@app.route('/files')
def get_files():
    media_files = sorted(os.listdir('media'))
    return jsonify({'files': media_files})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    _, file_extension = os.path.splitext(file.filename)
    if file and allowed_file(file.filename):
        filename = file.filename
        if custom_filename := request.form.get('fileName'):
            filename = f'{custom_filename}{file_extension}'
        filename = secure_filename(filename)
        if filename in os.listdir('media'):
            return jsonify({'status': 'error', 'message': f'File "{filename}" already exists, please choose another name'})
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'status': 'success', 'message': f'File "{filename}" uploaded successfully', 'filename': filename})
    return jsonify({'status': 'error', 'message': f'File type "{file_extension}" not allowed. Should be one of {ALLOWED_EXTENSIONS}'})


@app.route('/text', methods=['POST'])
def handle_text_video():
    text = request.form.get('text', 'Default Text')
    font_size = int(request.form.get('font_size') or 200)
    bg_color = request.form.get('bg_color') or '#000000'
    bg_color_rgb = hex_to_rgb(bg_color)
    bg_color_name = approximate_color_name(bg_color)
    text_color = request.form.get('text_color') or '#FFFFFF'
    text_color_name = approximate_color_name(text_color)
    sanitized_text = ''.join(c if c.isalnum() else '_' for c in text.lower())
    sanitized_text = sanitized_text[:30]  # Limit the length of the text part
    # Format the filename
    filename = f'__text__{sanitized_text}__{text_color_name}-on-{bg_color_name}__fs{font_size}px.mp4'
    try:
        create_scrolling_text_video(text, filename, bg_color_rgb, text_color, font_size)
        return jsonify({'status': 'success', 'message': f'Video "{filename}" created successfully', 'filename': filename})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def toggle_stop_video():
    global stop_video
    stop_video = True
    time.sleep(0.05)
    stop_video = False


@app.route('/play', methods=['POST'])
def play():
    global filename, stop_video, queue
    selected_file = request.form.get('file')
    if not selected_file:
        stop_video = False
        msg = 'Playing from queue...' if queue else 'Waiting for file...'
        return jsonify({'status': 'success', 'message': msg})
    if selected_file and selected_file in os.listdir('media'):
        filename = selected_file
        toggle_stop_video()
        return jsonify({'status': 'success', 'message': f'Playing {filename}...'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid file selected'})


@app.route('/queue/add', methods=['POST'])
def add_to_queue():
    global queue
    file = request.form.get('file')
    if file in os.listdir('media'):
        queue.append(file)
        return jsonify({'status': 'success', 'message': f'Added {file} to queue', 'queue': queue})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid file selected'})


@app.route('/queue/remove', methods=['POST'])
def remove_from_queue():
    file = request.form.get('file')
    if file in queue:
        queue.remove(file)
        return jsonify({'status': 'success', 'message': f'Removed {file} from queue', 'queue': queue})
    else:
        return jsonify({'status': 'error', 'message': 'File not in queue'})


@app.route('/queue', methods=['GET'])
def get_queue():
    global queue, loop_queue
    return jsonify({'queue': queue, 'loop': loop_queue})


@app.route('/playing', methods=['GET'])
def get_currently_playing_media():
    global filename
    return jsonify({'filename': filename or '-'})


@app.route('/set_speed', methods=['POST'])
def set_speed():
    global current_speed
    new_speed = request.form.get('speed', type=float)
    if new_speed is None:
        return jsonify({'status': 'error', 'message': 'Invalid speed value'})

    if new_speed <= 0:
        return jsonify({'status': 'error', 'message': 'Speed must be greater than 0'})

    current_speed = new_speed
    return jsonify({'status': 'success'})


@app.route('/get_speed', methods=['GET'])
def get_speed():
    global current_speed
    return jsonify({'speed': current_speed})



@app.route('/stop', methods=['POST'])
def stop():
    global stop_video
    stop_video = True
    return jsonify({'status': 'success', 'message': 'Playback stopped'})


@app.route('/playlist/save', methods=['POST'])
def save_playlist():
    global queue
    playlist_name = request.form.get('name')
    if not playlist_name:
        return jsonify({'status': 'error', 'message': 'Playlist name is required'})
    playlist_name = secure_filename(f'{playlist_name}.json')
    playlist_path = os.path.join(PLAYLIST_FOLDER, playlist_name)
    with open(playlist_path, 'w') as f:
        json.dump(queue, f)

    return jsonify({'status': 'success', 'message': f'Playlist "{playlist_name}" saved successfully'})


@app.route('/playlist/load', methods=['POST'])
def load_playlist():
    global queue
    playlist_name = request.form.get('name')
    if not playlist_name:
        return jsonify({'status': 'error', 'message': 'Playlist name is required'})

    playlist_path = os.path.join(PLAYLIST_FOLDER, f"{playlist_name}.json")
    if not os.path.exists(playlist_path):
        return jsonify({'status': 'error', 'message': f'Playlist "{playlist_name}" not found'})

    with open(playlist_path, 'r') as f:
        loaded_queue = json.load(f)

    # Filter out files that don't exist in the media folder
    queue = [file for file in loaded_queue if os.path.exists(os.path.join(UPLOAD_FOLDER, file))]

    return jsonify({'status': 'success', 'message': f'Playlist "{playlist_name}" loaded successfully', 'queue': queue})


@app.route('/queue/loop', methods=['POST'])
def toggle_loop():
    global loop_queue
    loop_queue = not loop_queue
    return jsonify({'status': 'success', 'loop': loop_queue})


@app.route('/playlists', methods=['GET'])
def get_playlists():
    playlists = [os.path.splitext(f)[0] for f in os.listdir(PLAYLIST_FOLDER) if f.endswith('.json')]
    return jsonify({'playlists': playlists})


@app.route('/media/<path:filename>')
def serve_sample(filename):
  return send_from_directory('media', filename)


def create_app():
    global app
    video_thread = threading.Thread(target=media_loop)
    video_thread.daemon = True
    video_thread.start()
    return app


if __name__ == '__main__':
    create_app()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)