import numpy as np
from core.template_parser.base_roi import BaseParser
from core.roi.roi_enums import DocumentType

class IDCardParser(BaseParser):

    # Function to decode the Egyptian ID number
    def parse_egyptian_id(self, id_number):
        try:
            governorates = {
                '01': 'Cairo',
                '02': 'Alexandria',
                '03': 'Port Said',
                '04': 'Suez',
                '11': 'Damietta',
                '12': 'Dakahlia',
                '13': 'Ash Sharqia',
                '14': 'Kaliobeya',
                '15': 'Kafr El - Sheikh',
                '16': 'Gharbia',
                '17': 'Monoufia',
                '18': 'El Beheira',
                '19': 'Ismailia',
                '21': 'Giza',
                '22': 'Beni Suef',
                '23': 'Fayoum',
                '24': 'El Menia',
                '25': 'Assiut',
                '26': 'Sohag',
                '27': 'Qena',
                '28': 'Aswan',
                '29': 'Luxor',
                '31': 'Red Sea',
                '32': 'New Valley',
                '33': 'Matrouh',
                '34': 'North Sinai',
                '35': 'South Sinai',
                '88': 'Foreign'
            }
            id_number = id_number.replace(" ", "")
            # Check if ID is valid and contains enough digits
            if id_number == "Unknown" or len(id_number) < 14:
                return {
                    DocumentType.BIRTH_DATE: "Unknown",
                    DocumentType.GOVERNORATE: "Unknown",
                    DocumentType.GENDER: "Unknown"
                }

            century_digit = int(id_number[0])
            year = int(id_number[1:3])
            month = int(id_number[3:5])
            day = int(id_number[5:7])
            governorate_code = id_number[7:9]
            gender_code = int(id_number[12:13])

            if century_digit == 2:
                century = "1900-1999"
                full_year = 1900 + year
            elif century_digit == 3:
                century = "2000-2099"
                full_year = 2000 + year
            else:
                raise ValueError("Invalid century digit")

            gender = "Male" if gender_code % 2 != 0 else "Female"
            governorate = governorates.get(governorate_code, "Unknown")
            birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

            return {
                DocumentType.BIRTH_DATE: birth_date,
                DocumentType.GOVERNORATE: governorate,
                DocumentType.GENDER: gender
            }
        except Exception as e:
            print(f"Error decoding ID: {e}")
            return {
                DocumentType.BIRTH_DATE: "Unknown",
                DocumentType.GOVERNORATE: "Unknown",
                DocumentType.GENDER: "Unknown"
            }
            
    def parse_egyption_demo(self, demo):
        demo_dict = {}
        if "ذكر" in demo:
            demo_dict[DocumentType.GENDER] = "ذكر"
        elif "أنثى" in demo: 
            demo_dict[DocumentType.GENDER] = "أثنى"
        
        if "مسلم" in demo:
            demo_dict[DocumentType.RELIGION] = 'مسلم'
        elif 'مسيحي' in demo:
            demo_dict[DocumentType.RELIGION] = 'مسيحي'
        
        if 'أعزب' in demo: 
            demo_dict[DocumentType.SOCIAL_STATE] = 'أعزب'
        elif 'متزوج' in demo:
            demo_dict[DocumentType.SOCIAL_STATE] = 'متزوج'
        return demo_dict
        
    def parse(self, id_card_data: dict[str, str]) -> list[dict[str, str]]:
        """
        Parse the image and extract text from the specified ROI.
        
        Returns:
            dict: A dictionary containing the extracted text from the image corresponding to the ROI Key.
        """
        id_card_parsed_data = {}
        for k, v in id_card_data.items():
            id_card_parsed_data[k] = v
            if k in ["nid", "nid_back"]:
                id_card_parsed_data.update(self.parse_egyptian_id(v))
            elif k in ["demo"]:
                id_card_parsed_data.update(self.parse_egyption_demo(v))
        return id_card_parsed_data