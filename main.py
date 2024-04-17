# app.py
from flask import Flask, request, render_template, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://user:user123@cluster0.z5xjddp.mongodb.net/")

# Access the database and collection containing the job data
db = client["database"]
job_data_collection = db["job_data"]

# Convert the MongoDB collection into a pandas DataFrame
job_data = pd.DataFrame(list(job_data_collection.find()))

# Vectorize job descriptions
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(job_data["description"])

# Access the database and collection containing the user data
users_collection = db["users"]

# Convert the MongoDB collection into a pandas DataFrame
users = pd.DataFrame(list(users_collection.find()))

# Function to check if the provided username and password are valid
def check_credentials(username, password):
    with users.iterrows() as rows:
        for row in rows:
            if row[1]["username"] == username and row[1]["password"] == password:
                return True
        return False

# ... rest of the code

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if check_credentials(username, password):
            # Authentication successful, redirect to index
            return redirect(url_for("index"))
        else:
            # Authentication failed, show error message
            error_message = "Invalid username or password"
            return render_template("login.html", error_message=error_message)
    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def index():
    return render_template("index.html")

#... rest of the code

@app.route("/recommend", methods=["POST"])
def recommend_skills_and_courses():
    job_title = request.form["job_title"]

    # Find job descriptions matching the provided job title
    matching_jobs = job_data[job_data["title"].str.contains(job_title, case=False)]

    if len(matching_jobs) == 0:
        return render_template("no_results.html", job_title=job_title)

    # Vectorize job descriptions
    job_descriptions = matching_jobs["description"]
    job_tfidf = tfidf_vectorizer.transform(job_descriptions)

    # Calculate cosine similarity between user input and job descriptions
    similarity_scores = cosine_similarity(job_tfidf, tfidf_matrix)

    # Get top matching job
    top_index = similarity_scores.argmax()
    top_job = job_data.iloc[top_index]

    # Extract recommended skills and courses from top job description
    recommended_skills = top_job["skills"].split(", ") if "skills" in top_job else []
    recommended_courses = top_job["courses"].split(", ") if "courses" in top_job else []
    course_links = top_job["course_links"].split(", ") if "course_links" in top_job else []

    course_data = zip(recommended_courses, course_links)

    return render_template("results.html", job_title=top_job["title"], skills=recommended_skills, courses_with_links=course_data)

#... rest of the code
if __name__ == "__main__":
    app.run(debug=True)

