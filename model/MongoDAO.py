# DAO -> Data Access Object
# Access -> CRUD (Create, Read, Update, Delete)

from pymongo import MongoClient

# MongoDB Connection
def conn_mongo():
    client = MongoClient('localhost', 27017)  # IP address, Port
    db = client['local']  # Allocation 'local' db
    collection = db.get_collection('movie')
    return collection

# Create review data(데이터 등록)
def add_review(data):
    collection = conn_mongo()
    collection.insert_one(data) # Data save

# Select review data(데이터 조회)
def get_reviews():
    pass

