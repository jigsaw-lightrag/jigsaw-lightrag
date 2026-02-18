from sqlalchemy import Column, Integer, DateTime, text, NVARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RequestSeq(Base):
    __tablename__ = "request_seq"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    req_id = Column(NVARCHAR(100), index=True, comment="req_id")
    content = Column(NVARCHAR, comment="content")
    # SEARCH NSEARCH GENERATE
    req_type = Column(NVARCHAR(20), comment="req_type")
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
    scenario = Column(NVARCHAR(50), comment="scenario")
