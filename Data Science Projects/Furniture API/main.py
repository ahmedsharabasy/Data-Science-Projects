from fastapi import FastAPI
import Related
import uvicorn

app = FastAPI()

data = Related.load_data()
similarity_matrix = Related.create_similarity_matrix(data)

@app.get("/related_products/{id}")
def related_products(id: int, num_products: int = 25):
    related_products = Related.get_related_products(data, similarity_matrix, id, num_products)
    return related_products




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1")#, reload=True, port=3306

 