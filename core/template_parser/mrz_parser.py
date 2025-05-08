import numpy as np
import os
from datetime import datetime
from core.template_parser.base_roi import BaseParser

class MRZParser(BaseParser):
    def _cleanse_roi(self, mrz_text):
        print("MRZ TEXT:", mrz_text)
        input_list = mrz_text.replace(" ", "").split("\n")
        # selection_length = next((len(item) for item in input_list if "<" in item and len(item) in {30, 36, 44}), None,)
        # selection_length = next((len(item) for item in input_list if "<" in item), None,)
        new_list = [item for item in input_list if "<" in item]
        if len(new_list) not in [2,3]:
            return ""
        # new_list = [item for item in input_list if len(item) >= selection_length]
        print("CLEANSED MRZ TEXT:", new_list)
        return "\n".join(new_list)

    def _get_final_checkdigit(self, input_string, input_type):
        if input_type == "TD3":
            return self._get_checkdigit(input_string[1][:10] + input_string[1][13:20] + input_string[1][21:43])
        elif input_type == "TD2":
            return self._get_checkdigit(input_string[1][:10] + input_string[1][13:20] + input_string[1][21:35])
        else:
            return self._get_checkdigit(input_string[0][5:] + input_string[1][:7] + input_string[1][8:15]
                                         + input_string[1][18:29])

    def _get_checkdigit(self, input_string):
        weights_pattern = [7, 3, 1]

        total = 0
        for i, char in enumerate(input_string):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord("A") + 10
            else:
                value = 0
            total += value * weights_pattern[i % len(weights_pattern)]

        check_digit = total % 10

        return str(check_digit)

    def _format_date(self, input_date):
        formatted_date = str(datetime.strptime(input_date, "%y%m%d").date())

        return formatted_date

    def _get_birth_date(self, birth_date_str, expiry_date_str):
        birth_year = int(birth_date_str[:4])
        expiry_year = int(expiry_date_str[:4])

        if expiry_year > birth_year:
            return birth_date_str
        adjusted_year = birth_year - 100

        return f"{adjusted_year}-{birth_date_str[5:]}"

    def _is_valid(self, image):
        if isinstance(image, str):
            return bool(os.path.isfile(image))
        elif isinstance(image, np.ndarray):
            return image.shape[-1] == 3
        
    def _parse_mrz(self, mrz_text, include_checkdigit=True):
        if not mrz_text:
            return {"status": "FAILURE", "status_message": "No MRZ detected"}
        mrz_lines = mrz_text.strip().split("\n")
        if len(mrz_lines) not in [2, 3]:
            return {"status": "FAILURE", "status_message": "Invalid MRZ format"}

        mrz_code_dict = {}
        if len(mrz_lines) == 2:
            if mrz_lines[1][-1] == '<':
                mrz_code_dict["mrz_type"] = "MRVB" if len(mrz_lines[0]) == 36 else "MRVA"
            else:
                mrz_code_dict["mrz_type"] = "TD2" if len(mrz_lines[0]) == 36 else "TD3"

            # Line 1
            mrz_code_dict["document_code"] = mrz_lines[0][:2].strip("<")

            mrz_code_dict["issuer_code"] = mrz_lines[0][2:5]
            if not mrz_code_dict["issuer_code"].isalpha():
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Invalid MRZ format"

            names = mrz_lines[0][5:].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

            # Line 2
            mrz_code_dict["document_number"] = mrz_lines[1][:9].replace("<", "")
            document_number_checkdigit = self._get_checkdigit(mrz_code_dict["document_number"])
            if document_number_checkdigit != mrz_lines[1][9]:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Document number checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["document_number_checkdigit"] = document_number_checkdigit

            mrz_code_dict["nationality_code"] = mrz_lines[1][10:13]
            if not mrz_code_dict["nationality_code"].isalpha():
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Invalid MRZ format"

            mrz_code_dict["birth_date"] = mrz_lines[1][13:19]
            birth_date_checkdigit = self._get_checkdigit(mrz_code_dict["birth_date"])
            if birth_date_checkdigit != mrz_lines[1][19]:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Date of birth checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["birth_date_checkdigit"] = birth_date_checkdigit
            mrz_code_dict["birth_date"] = self._format_date(mrz_code_dict["birth_date"])

            mrz_code_dict["gender"] = mrz_lines[1][20]

            mrz_code_dict["expiry_date"] = mrz_lines[1][21:27]
            expiry_date_checkdigit = self._get_checkdigit(mrz_code_dict["expiry_date"])
            if expiry_date_checkdigit != mrz_lines[1][27]:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Date of expiry checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["expiry_date_checkdigit"] = expiry_date_checkdigit
            mrz_code_dict["expiry_date"] = self._format_date(mrz_code_dict["expiry_date"])
            mrz_code_dict["birth_date"] = self._get_birth_date(mrz_code_dict["birth_date"], mrz_code_dict["expiry_date"])

            if mrz_code_dict["mrz_type"] == "TD2":
                mrz_code_dict["optional_data"] = mrz_lines[1][28:35].strip("<")
            elif mrz_code_dict["mrz_type"] == "TD3":
                mrz_code_dict["optional_data"] = mrz_lines[1][28:42].strip("<")
                optional_data_checkdigit = self._get_checkdigit(mrz_code_dict["optional_data"].strip("<"))
                if optional_data_checkdigit != mrz_lines[1][42]:
                    mrz_code_dict["status"] = "FAILURE"
                    mrz_code_dict["status_message"] = "Optional data checksum is not matching"
                if include_checkdigit:
                    mrz_code_dict["optional_data_checkdigit"] = optional_data_checkdigit
            elif mrz_code_dict["mrz_type"] == "MRVA":
                mrz_code_dict["optional_data"] = mrz_lines[1][28:44].strip("<")
            else:
                mrz_code_dict["optional_data"] = mrz_lines[1][28:36].strip("<")

            final_checkdigit = self._get_final_checkdigit(mrz_lines, mrz_code_dict["mrz_type"])
            if (mrz_lines[1][-1] != final_checkdigit
                    and mrz_code_dict["mrz_type"] not in ("MRVA", "MRVB")):
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Final checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["final_checkdigit"] = final_checkdigit
        else:
            mrz_code_dict["mrz_type"] = "TD1"

            # Line 1
            mrz_code_dict["document_code"] = mrz_lines[0][:2].strip("<")

            mrz_code_dict["issuer_code"] = mrz_lines[0][2:5]
            if not mrz_code_dict["issuer_code"].isalpha():
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Invalid MRZ format"

            mrz_code_dict["document_number"] = mrz_lines[0][5:14]
            document_number_checkdigit = self._get_checkdigit(mrz_code_dict["document_number"])
            if document_number_checkdigit != mrz_lines[0][14]:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Document number checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["document_number_checkdigit"] = document_number_checkdigit

            mrz_code_dict["optional_data_1"] = mrz_lines[0][15:].strip("<")

            # Line 2
            mrz_code_dict["birth_date"] = mrz_lines[1][:6]
            birth_date_checkdigit = self._get_checkdigit(mrz_code_dict["birth_date"])
            if birth_date_checkdigit != mrz_lines[1][6]:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Date of birth checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["birth_date_checkdigit"] = birth_date_checkdigit
            mrz_code_dict["birth_date"] = self._format_date(mrz_code_dict["birth_date"])

            mrz_code_dict["gender"] = mrz_lines[1][7]

            mrz_code_dict["expiry_date"] = mrz_lines[1][8:14]
            expiry_date_checkdigit = self._get_checkdigit(mrz_code_dict["expiry_date"])
            if expiry_date_checkdigit != mrz_lines[1][14]:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Date of expiry checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["expiry_date_checkdigit"] = expiry_date_checkdigit
            mrz_code_dict["expiry_date"] = self._format_date(mrz_code_dict["expiry_date"])

            mrz_code_dict["birth_date"] = self._get_birth_date(mrz_code_dict["birth_date"], mrz_code_dict["expiry_date"])

            mrz_code_dict["nationality_code"] = mrz_lines[1][15:18]
            if not mrz_code_dict["nationality_code"].isalpha():
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Invalid MRZ format"

            mrz_code_dict["optional_data_2"] = mrz_lines[0][18:29].strip("<")
            final_checkdigit = self._get_final_checkdigit(mrz_lines, mrz_code_dict["mrz_type"])
            if mrz_lines[1][-1] != final_checkdigit:
                mrz_code_dict["status"] = "FAILURE"
                mrz_code_dict["status_message"] = "Final checksum is not matching"
            if include_checkdigit:
                mrz_code_dict["final_checkdigit"] = final_checkdigit

            # Line 3
            names = mrz_lines[2].split("<<")
            mrz_code_dict["surname"] = names[0].replace("<", " ")
            mrz_code_dict["given_name"] = names[1].replace("<", " ")

        mrz_code_dict["mrz_text"] = mrz_text

        # Final status
        if mrz_code_dict.get("status") != "FAILURE":
            mrz_code_dict["status"] = "SUCCESS"

        return mrz_code_dict
    
    def parse(self, id_card_data: dict[str, str]) -> list[dict[str, str]]:
        """
        Parse the image and extract text from the specified ROI.
        
        Returns:
            dict: A dictionary containing the extracted text from the image corresponding to the ROI Key.
        """
        id_card_parsed_data = {}
        for k, v in id_card_data.items():
            id_card_parsed_data[k] = v
            if k in ["mrz"]:
                v = v.replace(" ", "")
                id_card_parsed_data.update(self._parse_mrz(v))
                
        return id_card_parsed_data