from datetime import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pfnopt.storages.rdb import models


def test_trial_model():
    # type: () -> None

    engine = create_engine('sqlite:///:memory:')
    session = Session(bind=engine)
    models.BaseModel.metadata.create_all(engine)

    datetime_1 = datetime.now()

    study = models.StudyModel('uuid')
    session.add(models.TrialModel(study))
    session.commit()

    datetime_2 = datetime.now()

    trial_model = session.query(models.TrialModel).first()
    assert datetime_1 < trial_model.datetime_start < datetime_2
    assert trial_model.datetime_complete is None


def test_version_info_model():
    # type: () -> None

    engine = create_engine('sqlite:///:memory:')
    session = Session(bind=engine)
    models.BaseModel.metadata.create_all(engine)

    session.add(models.VersionInfoModel())
    session.commit()

    # test check constraint of version_info_id
    version_info = models.VersionInfoModel()
    version_info.schema_version = 2
    version_info.library_version = '0.0.2'
    session.add(version_info)
    pytest.raises(IntegrityError, lambda: session.commit())
