from sqlalchemy import Column, Integer, DateTime, text, NVARCHAR
from sqlalchemy.orm import relationship
from .base import Base

class SamplingDatasetQA(Base):
    __tablename__ = "sampling_dataset_qa"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(NVARCHAR(200), index=True, comment="the txt filename of each dataset record content")
    question = Column(NVARCHAR(500), comment="question from dataset data")
    answer = Column(NVARCHAR(), comment="ground truth answer from dataset data")
    dataset = Column(NVARCHAR(50), comment="dataset")
    filelist = Column(NVARCHAR(500), comment="ground truth context filelist, for LongBench, PubMedQA, and QASPER, equals to filename")

    exp_results = relationship("QAExpResult", back_populates="standard_qa")
