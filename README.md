# Movie Recommendation System 🎬

A comprehensive implementation of multiple recommendation algorithms using the MovieLens 100k dataset, demonstrating popularity-based, content-based, and collaborative filtering approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Model Performance](#model-performance)
- [System Architecture](#system-architecture)
- [Customization](#customization)
- [File Structure](#file-structure)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Overview

This project implements a Netflix-like recommendation system with three distinct algorithms:

- **Popularity-Based Filtering**: Recommends highly-rated movies with sufficient rating counts
- **Content-Based Filtering**: Recommends movies similar to a given movie based on genre similarity  
- **Collaborative Filtering**: Uses matrix factorization (SVD) to predict user preferences

## ✨ Features

- **Automated Data Acquisition**: Downloads and processes the MovieLens 100k dataset automatically
- **Multiple Recommendation Strategies**: Three different recommendation approaches in one system
- **Model Evaluation**: Includes RMSE and MAE metrics for collaborative filtering performance
- **Production-Ready Architecture Guidance**: Conceptual framework for scaling to production systems

## 🔧 Requirements

Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn scikit-surprise requests
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scikit-learn` | TF-IDF vectorization and cosine similarity |
| `scikit-surprise` | Collaborative filtering algorithms |
| `requests` | HTTP library for dataset download |

## 📊 Dataset

The system uses the **MovieLens 100k dataset**, which contains:

- 100,000 ratings from 943 users on 1,682 movies
- User demographic information  
- Movie metadata including genres

> [!NOTE]
> The dataset is automatically downloaded on first run - no manual setup required!

## 🚀 Quick Start

1. **Clone or download the script**
2. **Install dependencies** (see [Requirements](#requirements))
3. **Run the system**:

```bash
python recommendation_system.py
```

The script will automatically:
- Download the MovieLens dataset
- Process and clean the data
- Train all three recommendation models
- Display sample recommendations

## 💻 Usage

### Getting Recommendations

#### 1. Popularity-Based Recommendations

```python
# Get top 10 popular movies
popular_recs = get_popular_recommendations(num_recommendations=10)
print(popular_recs)
```

**Output Example:**
```
['Schindler's List (1993)', 'Shawshank Redemption, The (1994)', ...]
```

#### 2. Content-Based Recommendations

```python
# Get movies similar to Star Wars
movie_title = "Star Wars (1977)"
content_recs = get_content_based_recommendations(
    movie_title, cosine_sim, movies_df, indices, num_recommendations=10
)
print(content_recs)
```

**Output Example:**
```
['Empire Strikes Back, The (1980)', 'Return of the Jedi (1983)', ...]
```

#### 3. Collaborative Filtering (SVD) Recommendations

```python
# Get personalized recommendations for user ID 1
user_id = 1
svd_recs = get_svd_recommendations(
    user_id, movies_df, ratings_df, svd_model, num_recommendations=10
)
print(svd_recs)
```

**Output Example:**
```
                    movie_title  movie_id
0  Casablanca (1942)            171
1  North by Northwest (1959)    408
...
```

## 🧠 Algorithm Details

### 1. Popularity-Based Filtering

- **Method**: Calculates average ratings and rating counts for each movie
- **Filter**: Movies with minimum threshold of ratings (default: 50)  
- **Ranking**: Movies ranked by average rating
- **Use Case**: Cold start problem for new users

### 2. Content-Based Filtering

- **Method**: Uses TF-IDF vectorization on movie genres
- **Similarity**: Calculates cosine similarity between movies
- **Recommendation**: Movies with highest similarity scores to input movie
- **Use Case**: When user preferences are known for specific movies

### 3. Collaborative Filtering (SVD)

- **Method**: Singular Value Decomposition matrix factorization
- **Hyperparameters**: 
  - 20 epochs
  - Learning rate: 0.005
  - Regularization: 0.02
- **Filtering**: Excludes already-rated movies for personalized recommendations
- **Use Case**: Personalized recommendations for active users

## 📈 Model Performance

The SVD model is evaluated using:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error - measures prediction accuracy |
| **MAE** | Mean Absolute Error - average prediction error |

> [!TIP]
> Lower RMSE and MAE values indicate better model performance.

## 🏗️ System Architecture

### Production Considerations

For a production Netflix-like system, consider implementing:

#### Core Components

1. **API Layer**
   ```python  
   # Example Flask API endpoint
   @app.route('/recommendations/<user_id>')
   def get_recommendations(user_id):
       if is_new_user(user_id):
           return get_popular_recommendations()
       else:
           return get_svd_recommendations(user_id)
   ```

2. **Model Serving**
   - Containerized model deployment with auto-scaling
   - Load balancing for high availability
   - Caching for frequently accessed recommendations

3. **Data Pipelines**
   - **Real-time**: Kafka for streaming user interactions
   - **Batch**: Apache Spark for periodic model retraining

4. **Hybrid Orchestration**
   - Smart routing between multiple algorithms
   - A/B testing framework for algorithm comparison

5. **Infrastructure**
   - **Databases**: User profiles, content metadata, interaction logs
   - **Caching**: Redis/Memcached for performance optimization
   - **Monitoring**: Model performance, latency, and engagement metrics

#### Cloud Services

| Provider | Services |
|----------|----------|
| **AWS** | SageMaker, Lambda, DynamoDB |
| **Google Cloud** | AI Platform, Cloud Functions, Firestore | 
| **Azure** | Machine Learning, Functions, Cosmos DB |

## ⚙️ Customization

### Hyperparameter Tuning

```python
# SVD model parameters
svd_model = SVD(
    n_epochs=20,      # Training epochs
    lr_all=0.005,     # Learning rate  
    reg_all=0.02,     # Regularization
    n_factors=100     # Latent factors
)
```

### Extending Content Features

Add more sophisticated content-based filtering:

- **Movie descriptions** (NLP with word embeddings)
- **Cast and crew information**
- **Release year and duration**
- **User demographic preferences**

```python
# Example: Adding movie description similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine genres and descriptions
movies_df['content_features'] = movies_df['genres_str'] + ' ' + movies_df['description']
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content_features'])
```

## 📁 File Structure

```
project/
├── recommendation_system.py    # Main implementation
├── ml-100k/                   # MovieLens dataset (auto-downloaded)
│   ├── u.data                 # User ratings data
│   ├── u.item                 # Movie metadata
│   ├── u.user                 # User demographic data
│   └── ...                    # Additional dataset files
├── README.md                  # This documentation
└── requirements.txt           # Python dependencies
```

## ⚠️ Limitations

> [!WARNING]
> Current implementation limitations to consider:

- **Scalability**: Single-machine implementation, not distributed
- **Cold Start**: Limited handling for completely new users/items
- **Real-time Updates**: No incremental learning capabilities  
- **Diversity**: May suffer from filter bubble effects
- **Bias**: No fairness considerations in recommendations

## 🚀 Future Enhancements

### Advanced Algorithms

- [ ] **Deep Learning**: Neural collaborative filtering, autoencoders
- [ ] **Multi-armed Bandits**: Dynamic exploration-exploitation balance
- [ ] **Graph Neural Networks**: Social network-based recommendations
- [ ] **Reinforcement Learning**: Long-term user engagement optimization

### System Improvements

- [ ] **Real-time Learning**: Online learning capabilities
- [ ] **Fairness & Bias**: Algorithmic fairness considerations
- [ ] **Explainability**: Recommendation explanation features
- [ ] **Multi-objective**: Balance relevance, diversity, and novelty

### Technical Enhancements

- [ ] **Microservices Architecture**: Service-oriented design
- [ ] **Kubernetes Deployment**: Container orchestration
- [ ] **Stream Processing**: Real-time data pipelines
- [ ] **Feature Store**: Centralized feature management

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -am 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/improvement`)  
5. **Create** a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 Movie Recommendation System

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

## 📚 References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/) - Research data from GroupLens
- [Surprise Documentation](https://surprise.readthedocs.io/) - Python library for recommender systems
- [Collaborative Filtering Survey](https://ieeexplore.ieee.org/document/7273729) - Comprehensive overview of CF techniques
- [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) - Netflix Prize insights

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the data science community

</div>