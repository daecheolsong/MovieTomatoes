# MongoDB Access and CRUD test
import sysconfig

from pymongo import MongoClient

# 1. MongoDB Connection
client = MongoClient('localhost', 27017)  # IP address, Port number
# IP --> Computer 구별 # Port --> 컴퓨터의 프로그램 구별 , 프로그램마다 port 다름
# localhost = 127.0.0.1 실제 내 ip
db = client['local']  # Allocating 'local' DB
collection = db.get_collection('test')  # Allocating 'review' Collection

data = {'name': 'cherry', 'age': 0}
collection.insert_one(data)
# MongoDB > database > collection > document

# CRUD -> Crete, Read, Update, Delete
