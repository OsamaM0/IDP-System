from enum import Enum

class DocumentType(str, Enum):
    MRZ = "mrz"
    NID = "nid"
    NID_BACK = "nid_back"
    DOB = "dob"
    LAST_NAME = "lastName"
    ADDRESS = "address"
    GOVERNORATE = "governorate"
    GENDER = "gender"
    SERIAL = "serial"
    JOB = "job"
    EXPIRY = "expiry"
    DEMO = "demo"
    RELIGION = "religion"
    SOCIAL_STATE = "social_state"
    ISSUE = "issue"
    MRZ_TYPE = "mrz_type"
    DOCUMENT_CODE = "document_code"
    ISSUER_CODE = "issuer_code"
    SURNAME = "surname"
    GIVEN_NAME = "given_name"
    DOCUMENT_NUMBER = "document_number"
    DOCUMENT_NUMBER_CHECKDIGIT = "document_number_checkdigit"
    NATIONALITY_CODE = "nationality_code"
    BIRTH_DATE = "birth_date"
    BIRTH_DATE_CHECKDIGIT = "birth_date_checkdigit"
    EXPIRY_DATE_CHECKDIGIT = "expiry_date_checkdigit"
    OPTIONAL_DATA = "optional_data"
   
    FINAL_CHECKDIGIT = "final_checkdigit"
    MRZ_TEXT = "mrz_text"
    PASSPORT = "passport"
    
    
    @staticmethod
    def get_all_classes():
        return [DocumentType.NIDB, DocumentType.NIDF, DocumentType.TAX, DocumentType.PASSPORT, DocumentType.DRIVING_LICENSE]
    
    @staticmethod
    def get_all_values():
        return [DocumentType.NIDB.value, DocumentType.NIDF.value, DocumentType.TAX.value, DocumentType.PASSPORT.value, DocumentType.DRIVING_LICENSE.value]
    
    @staticmethod
    def get_document_validator_values():
        return [DocumentType.NIDB.value, DocumentType.NIDF.value, DocumentType.TAX.value, DocumentType.PASSPORT.value]
        
    @staticmethod
    def get_all_classes_except(class_name):
        return [class_name for class_name in DocumentType.get_all_classes() if class_name != class_name]