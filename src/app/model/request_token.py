from sqlalchemy import Column, Integer, DateTime, text, NVARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RequestToken(Base):
    __tablename__ = "request_token"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    req_id = Column(NVARCHAR(100), index=True, comment="req_id")
    req_type = Column(NVARCHAR(20), comment="req_type")
    completion_tokens = Column(Integer, comment="completion_tokens")
    prompt_tokens = Column(Integer, comment="prompt_tokens")
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
