# Maternal Healthcare Backend

A Django REST Framework backend for the SHE-AI Qassim maternal healthcare platform.

## Features

- **Patient Management**: Complete patient records with medical history
- **AI Predictions**: Machine learning models for PPH, CS, and Neonatal outcome predictions
- **Chatbot**: AI-powered medical assistant (Dr. Noura)
- **RESTful API**: Full CRUD operations with authentication

## Tech Stack

- Django 5.2.6
- Django REST Framework 3.15.2
- PostgreSQL
- scikit-learn (ML models)
- pandas & numpy (data processing)

## Installation

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```env
SECRET_KEY=your-secret-key-here
DEBUG=True
DB_NAME=maternal_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
ALLOWED_HOSTS=localhost,127.0.0.1
```

4. **Run migrations:**
```bash
python manage.py migrate
```

5. **Create a superuser:**
```bash
python manage.py createsuperuser
```

6. **Load initial data (optional):**
```bash
python manage.py load_medical_qa
```

7. **Run the development server:**
```bash
python manage.py runserver 8003
```

## API Endpoints

### Authentication
- `POST /api/auth/register/` - Register new user
- `POST /api/auth/login/` - Login
- `POST /api/auth/logout/` - Logout

### Patients
- `GET /api/patients/` - List all patients
- `POST /api/patients/` - Create new patient
- `GET /api/patients/{id}/` - Get patient details
- `PUT /api/patients/{id}/` - Update patient
- `DELETE /api/patients/{id}/` - Delete patient

### AI Predictions
- `GET /api/patients/{id}/predict_pph/` - PPH prediction
- `GET /api/patients/{id}/predict_neonatal/` - Neonatal outcome prediction
- `POST /api/predict-cs/` - Cesarean section prediction

### Chatbot
- `POST /api/chatbot/chat/` - Chat with Dr. Noura

## Project Structure

```
maternal_backend/
├── maternal_backend/      # Main project settings
├── patients/              # Patient management app
├── bot/                   # Chatbot app
├── chatbot_ai/            # AI chatbot backend
├── artifacts/             # ML model files
└── manage.py
```

## Machine Learning Models

The project includes pre-trained models for:
- **PPH Prediction**: Postpartum hemorrhage risk assessment
- **CS Prediction**: Cesarean section likelihood
- **Neonatal Outcome**: Neonatal health prediction

Models are stored in the `artifacts/` directory.

## Development

Run tests:
```bash
python manage.py test
```

Create migrations after model changes:
```bash
python manage.py makemigrations
python manage.py migrate
```

## License

© 2024 Qassim Health Cluster. All rights reserved.
