import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# def load_data():
#     cnx = mysql.connector.connect(
#         host='https://college.makkah-tech.com/',
#         password='ahmed-000',
#         user='ahmed_000',
#         database="college_eslam",
#     )

# def load_data():
#     cnx = mysql.connector.connect(
#         host='127.0.0.1',
#         password='789mn321kjoi',
#         user='root',
#         database="college_eslam",
#     )

# def load_data():
#     cnx = mysql.connector.connect(
#         host='127.0.0.1',
#         password='ahmed-000',
#         user='ahmed_000',
#         database="college_eslam",
#     )

#     engine = create_engine('mysql+mysqlconnector://ahmed_000:ahmed-000@127.0.0.1/college_eslam')
#     query = "SELECT * FROM products"
#     df = pd.read_sql_query(query, con=cnx)


def load_data():
    cnx = mysql.connector.connect(
        host='127.0.0.1',
        password='ahmed-000',
        user='ahmed_000',
        database="college_eslam",
        use_pure=True
        # ssl_ca='/path/to/ssl/ca/file.pem' # add this parameter to enable SSL encryption
    )

    engine = create_engine('mysql+mysqlconnector://ahmed_000:ahmed-000@127.0.0.1/college_eslam')
    # engine = create_engine('mysql+mysqlconnector://root:789mn321kjoi@127.0.0.1/college_eslam') 
    query = "SELECT * FROM products"
    df = pd.read_sql_query(query, con=cnx)

    reviews_query = "SELECT product_id, CEIL(AVG(star)) as star_mean, COUNT(*) as review_count FROM reviews GROUP BY product_id"
    reviews_df = pd.read_sql_query(reviews_query, con=cnx)
    df = pd.merge(df, reviews_df, how='left', left_on='id', right_on='product_id')
    df = df.set_index('id')
    
    media_query = "SELECT id, model_type, model_id, uuid, collection_name, name, file_name, mime_type, disk, conversions_disk FROM media WHERE model_type='App\\\\Models\\\\Product' AND collection_name='image'"
    # media_query = f"SELECT * FROM media WHERE model_type='App\Models\Product' LIMIT 1;"
    media_df = pd.read_sql_query(media_query, con=cnx)
    media_df = media_df.set_index('model_id')
    df = df.join(media_df)
    # df['image_url'] = f"{df['disk']}/{df['file_name']}"
    # df['image_url'] = pd.concat([df['disk'], df['file_name']], axis=1)
    # df['image_url'] = "https://college.makkah-tech.com/public/storage" + df['disk'] +'/'+df['ids']+'/'+df['file_name']
    df['id'] = df['id'].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    df['image_url'] = "https://college.makkah-tech.com/public/storage/" + df['disk'].astype(str) + '/' + df['id'].astype(str) + '/' + df['file_name'].astype(str)
    print(df)

    cnx.close()
    return df


def create_similarity_matrix(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer(stop_words='english')

    product_descriptions = df['description']
    product_categories = df['category']
    product_names = df['title']

    tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)
    category_matrix_weight = 2
    count_matrix_category = count_vectorizer.fit_transform(category_matrix_weight * product_categories)
    count_matrix_name = count_vectorizer.fit_transform(product_names)
    count_matrix = hstack([count_matrix_category, count_matrix_name])

    combined_similarity_matrix = cosine_similarity(hstack([tfidf_matrix, count_matrix]))
    return combined_similarity_matrix


def get_related_products(df, similarity_matrix, id, num_products=25):
    if id not in df.index:
        return {"error": f"{id} not found in product list."}
    idx = df.index.get_loc(id)
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_products+1]]

    related_products = df.iloc[sim_indices][['title','newPrice','oldPrice','offer','star_mean','review_count','uuid','image_url']]
    related_products = related_products.fillna(0)
    related_products.insert(0, 'id', related_products.index)
    if len(related_products) == 0:
        return {"error": "No related products found."}

    return {"id": id, "Related Products": related_products.to_dict(orient="records")}

