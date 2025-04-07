from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
import pymysql
import bcrypt

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Replace with your MySQL username
    'password': 'MySqlR00t@2023!',  # Replace with your MySQL password
    'database': 'users_driverd',
    'cursorclass': pymysql.cursors.DictCursor
}

# Helper function to connect to the database
def get_db_connection():
    try:
        return pymysql.connect(**db_config)
    except pymysql.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Serve HTML pages from public folder
@app.route('/')
def serve_index():
    return send_from_directory('../public', 'register.html')

@app.route('/login')
def serve_login():
    return send_from_directory('../public', 'login.html')

# Serve static CSS files correctly
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('../public/css', filename)

# Serve dashboard page after login
@app.route('/dashboard')
def serve_dashboard():
    return send_from_directory('../public', 'dashboard.html')

# Signup endpoint
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO usersinfoddd (username, email, password) VALUES (%s, %s, %s)"
            cursor.execute(sql, (username, email, hashed_password))
            connection.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except pymysql.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM usersinfoddd WHERE email = %s"
            cursor.execute(sql, (email,))
            user = cursor.fetchone()

            if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                # Redirect to dashboard on successful login
                return redirect(url_for('serve_dashboard'))
            else:
                return jsonify({'error': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
