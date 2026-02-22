╔═══════════════════════════════════════════════════════════════╗
║                      OptiFrameY Project                       ║
╚═══════════════════════════════════════════════════════════════╝

  Python Version: 3.11.0


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PREREQUISITES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Before getting started, make sure:

    • Python 3.11.0 is installed on your system.
    • You are in the project root folder (OptiFrameY/) for all commands.
    • The virtual environment is activated before running anything.

  To activate the virtual environment (Windows):

    myenv\Scripts\activate


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. BACKEND SETUP & RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1 — Activate the virtual environment (from project root):

    myenv\Scripts\activate

  Step 2 — Navigate to the backend directory:

    cd backend

  Step 3 — Install dependencies:

    pip install -r requirements.txt

  Step 4 — Apply database migrations:

    python manage.py makemigrations
    python manage.py migrate

  Step 5 — Create a superuser (for admin access):

    python manage.py createsuperuser

    You will be prompted to enter:
      • Username
      • Email address
      • Password

  Step 6 — Start the backend server:

    python manage.py runserver

  The backend will be available at:
    → API:        http://127.0.0.1:8000/
    → Admin panel: http://127.0.0.1:8000/admin/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. CELERY WORKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  In a separate terminal, activate the virtual environment, then
  navigate to the backend directory and run:

    celery -A backend worker --loglevel=info -P solo

  NOTE: The Celery worker must be running alongside the backend
  server for task queue functionality to work correctly.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. FRONTEND SETUP & RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1 — Activate the virtual environment (from project root):

    myenv\Scripts\activate

  Step 2 — Navigate to the frontend directory:

    cd frontend

  Step 3 — Launch the Streamlit app:

    streamlit run app.py

  NOTE: The backend server must be running before using any
  features that depend on the API.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AUTHENTICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The API uses Token Authentication via Django REST Framework.

    • Sign in from the Streamlit app using the username and password
      of a superuser created with `python manage.py createsuperuser`.

    • After login, the frontend automatically attaches the token to
      every API request via the Authorization header:

        Authorization: Token <your_token>

    • Use the "Log out" option in the sidebar to clear the session.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUICK REFERENCE — STARTUP ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  For a full working environment, start services in this order:

    [Terminal 1]  Backend server   →  python manage.py runserver
    [Terminal 2]  Celery worker    →  celery -A backend worker --loglevel=info -P solo
    [Terminal 3]  Frontend app     →  streamlit run app.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━