OptiFrameY Project
=================

Python Version
--------------
Python 3.11.0


=====================
HOW TO RUN BACKEND
=====================

1. Open the project root folder:
   OptiFrameY

2. Activate the virtual environment:
   myenv/Scripts/activate

3. Navigate to the backend directory:
   cd backend

4. Install required dependencies:
   pip install -r requirements.txt

5. Create a superuser (for admin access):
   python manage.py createsuperuser

   You will be prompted to enter:
   - Username
   - Email
   - Password

6. Apply database migrations:
   python manage.py makemigrations
   python manage.py migrate

7. Run the backend server:
   python manage.py runserver

8. Backend will be available at:
   http://127.0.0.1:8000/

9. Admin panel can be accessed at:
   http://127.0.0.1:8000/admin/


=====================
HOW TO RUN FRONTEND
=====================

1. Open the project root folder:
   OptiFrameY

2. Activate the virtual environment:
   myenv/Scripts/activate

3. Navigate to the frontend directory:
   cd frontend

4. Run the Streamlit application:
   streamlit run app.py


=====================
NOTES
=====================
- Ensure Python 3.11.0 is installed before running the project.
- Make sure the virtual environment is activated before running any commands.
- Backend must be running for frontend features that depend on APIs.
