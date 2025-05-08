import re 

class TextUtils:
    @staticmethod
    def fix_arabic_text(text):
        if not text:
            return text
        try:
            # Check if there are Arabic characters in the text
            if any('\u0600' <= c <= '\u06FF' for c in text):
                print(text)
                # Reshape the Arabic text
                s_text = [t[::-1] for t in text.split(" ")]
                s_text.reverse()
                text = " ".join(s_text)
            return text
        except:
            # If there's any error, return the original text
            return text
    
    @staticmethod
    # Function to remove numbers from a string
    def remove_numbers(text):
        return re.sub(r'\d+', '', text)