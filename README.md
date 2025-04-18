# 🛣️ FixMyRoad - Pothole Detection & Reporting System 🚧

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-red.svg)
![MongoDB](https://img.shields.io/badge/mongodb-atlas-green.svg)

## 📋 Overview

FixMyRoad is an intelligent pothole detection and reporting system that uses machine learning to identify, verify, and prioritize road damage issues. The platform allows citizens to report road problems while providing authorities with data-driven insights to efficiently allocate repair resources based on damage severity.

## ✨ Key Features

- **🔍 AI-Powered Pothole Detection**: Automatically validates and classifies road damage from user-submitted photos
- **📊 Smart Prioritization Algorithm**: Ranks issues based on damage severity, community votes, and other factors
- **🗺️ Interactive Mapping**: Displays verified potholes on an interactive map for easy location tracking
- **🔄 Community Engagement**: Allows users to upvote/downvote issues to improve prioritization
- **👨‍💻 Admin Dashboard**: Provides authorities with insights and tools to manage reported issues
- **🌉 Bridge Health Prediction**: Estimates bridge lifespan based on various structural parameters

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Database**: MongoDB Atlas
- **ML Integration**: Roboflow API (pothole detection)
- **Computer Vision**: OpenCV, Supervision
- **Data Processing**: NumPy, scikit-learn
- **Frontend**: HTML/CSS/JavaScript, Bootstrap
- **Visualization**: Interactive maps and data visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB Atlas account
- Roboflow API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anujsoni3/fixmyroad.git
   cd fixmyroad
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # MongoDB connection string
   export MONGODB_URI="your_mongodb_atlas_connection_string"
   
   # Roboflow API key
   export ROBOFLOW_API_KEY="your_roboflow_api_key"
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the application at http://localhost:5000

## 📱 Usage

### For Citizens

1. **Create an Account**: Register with your email and personal details
2. **Report Potholes**: Take a photo of road damage and submit with location details
3. **Track Status**: Monitor the status of your submitted reports
4. **Community Engagement**: Upvote/downvote reports to improve prioritization
5. **View Map**: See all reported issues on an interactive map

### For Administrators

1. **Login**: Use admin credentials to access the dashboard
2. **Manage Reports**: View, approve, reject, or mark issues as resolved
3. **Prioritize Repairs**: Use the AI-generated priority scores to plan maintenance
4. **Bridge Analysis**: Predict bridge lifespans based on structural parameters

## 📊 How It Works

1. **Image Upload**: User uploads a photo of a pothole with location data
2. **AI Verification**: Our ML model analyzes the image to detect potholes
3. **Damage Assessment**: System calculates severity based on size and number of potholes
4. **Priority Calculation**: Algorithm combines damage severity, community votes, and other factors
5. **Notification**: Authorities receive prioritized lists of issues for efficient resolution

## 🔒 Security Features

- Password hashing with Bcrypt
- Secure session management
- Input validation to prevent injection attacks
- Authorized access controls for admin features

## 📁 Project Structure

```
fixmyroad/
├── app.py                       # Main application file
├── requirements.txt             # Dependencies
├── models/                      # ML model files for bridge analysis
├── static/                      # Static assets (CSS, JS, images)
└── templates/                   # HTML templates
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Roboflow for providing the object detection API
- MongoDB Atlas for database services
- All contributors who have helped shape this project

## 🚀 Live Demo

[![Deploy](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge&logo=vercel)](https://fixmyroad-65da.onrender.com/)


## 📞 Contact

Anuj Soni - [GitHub](https://github.com/anujsoni3) - soni3anuj@gmail.com

Project Link: [https://github.com/anujsoni3/fixmyroad](https://github.com/anujsoni3/fixmyroad)
