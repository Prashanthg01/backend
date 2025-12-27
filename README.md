OptiFrameY Project
=================

Python Version
--------------
Python 3.11.0


=================================
HOW TO RUN THE BACKEND (Django)
=================================

1. Open the project root folder:
   OptiFrameY

2. Activate the virtual environment:
   myenv/Scripts/activate

3. Navigate to the backend directory:
   cd backend

4. Install required dependencies:
   pip install -r requirements.txt

5. Create a Django superuser:
   python manage.py createsuperuser

   You will be prompted to enter:
   - Username
   - Email
   - Password

6. Run the backend server:
   python manage.py runserver

7. Backend will be available at:
   http://127.0.0.1:8000/

8. Access the Django Admin Panel at:
   http://127.0.0.1:8000/admin/


=================================
HOW TO RUN THE FRONTEND (Streamlit)
=================================

1. Open the project root folder:
   OptiFrameY

2. Activate the virtual environment:
   myenv/Scripts/activate

3. Navigate to the frontend directory:
   cd frontend

4. Run the Streamlit application:
   streamlit run app.py


=================================
NOTES
=================================
- Ensure Python 3.11.0 is installed before running the project.
- Always activate the virtual environment before running backend or frontend services.
- Make sure all dependencies are installed successfully to avoid runtime errors.
