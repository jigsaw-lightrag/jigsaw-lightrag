from sqlalchemy import Column, Integer, DateTime, text, NVARCHAR, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class QAExpResult(Base):
    __tablename__ = "qa_exp_result"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    qa_id = Column(Integer, ForeignKey('sampling_dataset_qa.id'), comment="qa_id")
    actual_answer = Column(NVARCHAR(), comment="actual_answer")
    actual_filelist = Column(NVARCHAR(500), comment="actual filelist")
    create_at = Column(
        DateTime(),
        server_default=text("CURRENT_TIMESTAMP"),
        comment="create_at",
    )
    score = Column(Float, comment="evaluation score")
    report = Column(Text, comment="evaluation report")
    scenario = Column(NVARCHAR(50), comment="scenario")

    standard_qa = relationship(
        "SamplingDatasetQA",
        back_populates="exp_results",
        uselist=False
    )
