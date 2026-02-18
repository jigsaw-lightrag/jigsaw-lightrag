from sqlalchemy import Column, Integer, DateTime, text, NVARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Saving all documents' New, Modified, Persistent, Deleted lifecycle status in your DB table, which is the projection of your dataset raw files and subgraph pool.
class SubgraphPoolMapping(Base):
    __tablename__ = "subgraph_pool_mapping"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(NVARCHAR(200), index=True, comment="filename")  # identify
    md5 = Column(NVARCHAR(32), nullable=True, comment="md5")  
    create_at = Column(
        DateTime(),
        server_default=text("CURRENT_TIMESTAMP"),
        comment="create_at",
    )
    update_at = Column(
        DateTime(),
        server_default=text("CURRENT_TIMESTAMP"),
        comment="update_at",
    )
    filepath = Column(NVARCHAR(1000), nullable=True, comment="Document file path")
    cur_status = Column(NVARCHAR(20), nullable=True, comment="Current lifecycle status of document") 
    base_entry = Column(NVARCHAR(50), nullable=True, comment="Base entry of your KG data")
