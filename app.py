from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this for security

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'diabetic_retinopathy_db'

mysql = MySQL(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load the trained model
try:
    model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Class Labels
class_labels = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Non-proliferative Diabetic Retinopathy',
    2: 'Moderate Non-proliferative Diabetic Retinopathy',
    3: 'Severe Non-proliferative Diabetic Retinopathy',
    4: 'Proliferative Diabetic Retinopathy'
}

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, email FROM users WHERE id = %s", (user_id,))
        user_data = cur.fetchone()
        cur.close()
        if user_data:
            return User(id=user_data[0], username=user_data[1], email=user_data[2])
    except Exception as e:
        print(f"❌ Database Error: {e}")
    return None

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict the class of an image
def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Unable to read the image at {image_path}")
            return None

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = int(np.argmax(prediction, axis=1)[0])  # Convert NumPy int64 to Python int
        return predicted_class
    except Exception as e:
        print(f"❌ Error during image processing: {e}")
        return None

# Home Route → Redirects to dashboard if logged in, otherwise to login page
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                    (username, email, hashed_password))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user_data = cur.fetchone()
        cur.close()

        if user_data and bcrypt.check_password_hash(user_data[3], password):
            user = User(id=user_data[0], username=user_data[1], email=user_data[2])
            login_user(user, remember=True)  # Persistent session
            print(f"✅ User logged in: {user.username}, is_authenticated: {current_user.is_authenticated}")
            
            # 🔹 FIXED: Redirecting to `index.html` instead of `dashboard`
            return redirect(url_for('index'))  
        else:
            flash("Invalid email or password!", "danger")
    
    return render_template('login.html')

# Index Route → Where Prediction Takes Place (Added this Route)
@app.route('/index')
@login_required
def index():
    return render_template('index.html', username=current_user.username)

# Dashboard (Only for Navigation, No Prediction Here)
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

# Prediction Route
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static/images', filename)
            file.save(filepath)

            print(f"✅ Image saved at: {filepath}")

            predicted_class = predict_image(filepath)
            print(f"✅ Predicted class: {predicted_class}")

            if predicted_class is None:
                return jsonify({'error': 'Prediction failed. Check logs for details.'}), 500

            predicted_label = class_labels.get(predicted_class, 'Unknown')

            return jsonify({'predicted_class': predicted_class, 'predicted_label': predicted_label, 'filepath': filepath})

        return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        print(f"❌ Error in prediction route: {e}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Fully clear session data
    flash('Logged out successfully!', 'success')
    return redirect(url_for('dashboard'))

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
