from enum import Enum

class DocumentType(str, Enum):
    NIDF = "ID_Front"
    NIDB = "ID_Back"
    PASSPORT = "Passport"
    CARD = "Card"
    TAX = "Tax_Card"
    CHEQUE = "Cheque"
    VEHICLE_PLATE = "Car_Plate"
    RECEIPT = "Receipt"
    FORM = "form"
    TABLE = "table"
    TEXT = "text"
    NO_CLASS = "No_Class"
    
    @staticmethod
    def get_all_classes():
        return [DocumentType.NIDB, DocumentType.NIDF, DocumentType.TAX, DocumentType.PASSPORT, DocumentType.CARD, DocumentType.CHEQUE, DocumentType.VEHICLE_PLATE, DocumentType.RECEIPT, DocumentType.FORM, DocumentType.TABLE, DocumentType.TEXT]
    
    @staticmethod
    def get_all_values():
        return [doc_type.value for doc_type in DocumentType.get_all_classes()]
    
    @staticmethod
    def get_document_validator_values():
        return [DocumentType.NIDB.value, DocumentType.NIDF.value, DocumentType.TAX.value, DocumentType.PASSPORT.value, DocumentType.CARD.value, DocumentType.CHEQUE.value, DocumentType.VEHICLE_PLATE.value, DocumentType.RECEIPT.value]
        
    @staticmethod
    def get_all_classes_except(class_name):
        return [class_name for class_name in DocumentType.get_all_classes() if class_name != class_name]
    
    @staticmethod
    def get_document_type_by_value(value: str) -> str:
        """
        Get the document type by its value.
        """
        for doc_type in DocumentType:
            if doc_type.value == value:
                return doc_type
        return None