from enum import Enum
from config.config import get_settings
class ModelType(str, Enum):
    ID_DETECTOR_MODEL = "id_detector"
    MRZ_DETECTOR_MODEL = "mrz_detector"
    ID_PARTS_DETECTOR_MODEL = "id_parts_detector"
    ID_NUMBER_DETECTOR_MODEL = "id_number_detector"
    DOCUMENT_VERIFIER_MODEL = "document_verifier"
    
    @classmethod
    def get_all_classes(cls):
        return [cls.ID_DETECTOR_MODEL, cls.MRZ_DETECTOR_MODEL, cls.ID_PARTS_DETECTOR_MODEL, cls.ID_NUMBER_DETECTOR_MODEL]
    
    @classmethod
    def get_all_values(cls):
        return [model.value for model in cls.get_all_classes()]

    @classmethod
    def get_all_classes_except(cls, class_to_exclude):
        return [model_class for model_class in cls.get_all_classes() if model_class != class_to_exclude]

    @classmethod
    def get_path(cls, model_enum: 'ModelType') -> str:
        settings = get_settings()
        model_paths = {
            cls.ID_DETECTOR_MODEL: settings.ID_DETECTOR_MODEL_PATH,
            cls.MRZ_DETECTOR_MODEL: settings.MRZ_DETECTOR_MODEL_PATH,
            cls.ID_PARTS_DETECTOR_MODEL: settings.ID_PARTS_DETECTOR_MODEL_PATH,
            cls.ID_NUMBER_DETECTOR_MODEL: settings.ID_NUMBER_DETECTOR_MODEL_PATH,
            cls.DOCUMENT_VERIFIER_MODEL: settings.DOCUMENT_CLASSIFIER_MODEL_PATH,
        }
        return model_paths.get(model_enum, None)
