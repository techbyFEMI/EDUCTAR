from Database.educt_db import Base
from sqlalchemy import Column, String, Text, text
class markdownFiles(Base):
    __tablename__ = "markdown_files"
    
    file_path = Column(String, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    content = Column(Text)