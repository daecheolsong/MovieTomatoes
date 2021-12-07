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

    collection = conn_mongo() # MongoDB Connection
    review_list = []
    for one in collection.find({},{'_id':0,'title':1,'review':1,'score':1}): # 제목,리뷰 ,평점만 DB 에서 조회
        review_list.append([one['title'],one['review'],one['score']])
    return review_list

