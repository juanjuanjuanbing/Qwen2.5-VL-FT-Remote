# instruction_processor.py (Based on Qwen2.5, adapted to BBoxDetector's calling style)
from typing import Dict, Tuple
from openai import OpenAI
import re


class InstructionProcessor:
    def __init__(
        self, 
        model_api_base: str = "http://localhost:8000/v1",  # Qwen2.5 API address
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",  # Model name
        confidence_threshold: float = 80  # Confidence threshold for decomposition
    ):
        """
        Initialize instruction processor (using Qwen2.5 for classification and decomposition)
        Adapted to BBoxDetector's client management method, using context manager
        """
        self.config = {
            "api_key": "EMPTY",
            "api_base": model_api_base,
            "model_name": model_name,
            "temperature": 0.1,  # 0.0 for classification, 0.1 for decomposition
            "timeout": 60
        }
        self.confidence_threshold = confidence_threshold
        self.client = None  # Lazy initialization of client, created via context manager

    def __enter__(self):
        """Create OpenAI client (consistent with BBoxDetector)"""
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"]
        )
        return self

    def __exit__(self, *args):
        """Close client connection"""
        if self.client:
            self.client.close()
        self.client = None

    def _classify_with_qwen(self, instruction: str) -> Tuple[str, str, float]:
        """Call Qwen2.5 to classify the instruction: return (intent type, difficulty level, confidence score)"""
        if not self.client:
            raise RuntimeError("InstructionProcessor must be used with 'with' statement")

        prompt = f"""
        Task: Analyze the following aerial image detection instruction and make three judgments:
        1. Intent type (choose one of three):
           - Explicit: Clearly specifies the detection target (e.g., "Detect trucks in the image")
           - Implicit: Doesn't directly state the target but can be inferred (e.g., "Detect vehicles capable of transporting goods")
           - Ambiguous: Target is unclear (e.g., "Detect important objects")
        2. Difficulty level (choose one of two):
           - Simple: Short sentence (≤10 words), no complex modifiers
           - Complex: Long sentence (>10 words), or contains multiple conditions/modifiers (e.g., "Detect damaged residential buildings with windows in mountainous landslide disaster areas")
        3. Confidence score (0-100): 
           - A numerical value representing your certainty about the above two classifications
           - Higher value means higher confidence (80-100: confident, 50-79: moderate, <50: low confidence)
        
        Instruction: {instruction}
        
        Requirement: Only return the classification result in the format "intent type,difficulty level,confidence score" (e.g., "Explicit,Simple,95"), no other content.
        The confidence score must be an integer between 0 and 100.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Classification requires certainty, zero randomness
                max_tokens=20,
                timeout=self.config["timeout"]
            )
            result = response.choices[0].message.content.strip()
            
            # Parse result with fault tolerance
            parts = result.split(",")
            if len(parts) >= 3:
                intent = parts[0].strip()
                difficulty = parts[1].strip()
                # Extract confidence score with regex to handle non-numeric characters
                confidence_str = re.findall(r'\d+', parts[2])
                confidence = float(confidence_str[0]) if confidence_str else 0.0
                # Validate confidence range
                confidence = max(0.0, min(100.0, confidence))
                # Validate intent and difficulty values
                valid_intents = ["Explicit", "Implicit", "Ambiguous"]
                valid_difficulties = ["Simple", "Complex"]
                intent = intent if intent in valid_intents else "Ambiguous"
                difficulty = difficulty if difficulty in valid_difficulties else "Complex"
                return intent, difficulty, confidence
            else:
                return "Ambiguous", "Complex", 0.0  # Fallback for format errors
        except Exception as e:
            print(f"[Qwen classification failed] Error: {str(e)}, using default classification")
            return "Ambiguous", "Complex", 0.0

    def classify_intent(self, instruction: str) -> Tuple[str, float]:
        """Predict intent type using Qwen2.5 with confidence score"""
        intent, _, confidence = self._classify_with_qwen(instruction)
        return intent, confidence

    def evaluate_difficulty(self, instruction: str) -> Tuple[str, float]:
        """Predict difficulty level using Qwen2.5 with confidence score"""
        _, difficulty, confidence = self._classify_with_qwen(instruction)
        return difficulty, confidence

    def decompose_instruction(self, raw_instruction: str) -> str:
        """Decompose complex instructions into target objects and their environments using Qwen2.5"""
        if not self.client:
            raise RuntimeError("InstructionProcessor must be used with 'with' statement")

        prompt = f"""
        Task: Decompose the following aerial image detection instruction into two parts: 
        "target objects (to be detected)" and "their environments (background contexts)".
        Think step by step, but only output the two parts in the format "Targets: [list of objects]; Environment: [environment description]" without any extra information.
    
        Rules:
        1. Target objects: Specific entities that need detection (e.g., vehicles, buildings, ships), suitable for aerial scene recognition
        2. Environment: Background scenes where target objects are located (e.g., highways, ports, urban areas, farmland), used for auxiliary positioning
        3. Avoid abstract concepts; ensure both parts are observable concrete elements
        4. Use "none" if a part does not exist
    
        Original instruction: {raw_instruction}
        Decomposition result:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=100,
                timeout=self.config["timeout"]
            )
            processed_result = response.choices[0].message.content.strip()
            if "Targets: none" in processed_result and "Environment: none" in processed_result:
                return raw_instruction
            return processed_result
        except Exception as e:
            print(f"[Instruction decomposition failed] Error: {str(e)}, using original instruction as fallback")
            return raw_instruction

    def process(self, instruction: str) -> Tuple[Dict, str]:
        """Complete processing flow: Qwen classification → dynamic routing → return processed instruction"""
        # 1. Qwen classification and evaluation with confidence
        intent, difficulty, confidence = self._classify_with_qwen(instruction)
        process_result = {
            "raw_instruction": instruction,
            "intent": intent,
            "difficulty": difficulty,
            "confidence_score": confidence,
            "confidence_threshold": self.confidence_threshold,
            "classifier": "Qwen2.5"
        }
        print(process_result)

        # 2. Dynamic routing with confidence threshold
        # Only decompose if confidence > threshold AND (intent is not Explicit OR difficulty is not Simple)
        if (intent == "Explicit" and difficulty == "complex") or (confidence < self.confidence_threshold):
            final_instruction = instruction
            process_result["decomposed"] = False
            process_result["decomposition_reason"] = "No decomposition needed"
            final_instruction = self.decompose_instruction(instruction)
            process_result["decomposed"] = True
            process_result["decomposition_reason"] = "Decomposed (complex/implicit/ambiguous instruction with high confidence)"

        return process_result, final_instruction

# Usage example
if __name__ == "__main__":
    # Initialize processor with confidence threshold (90 by default)
    processor = InstructionProcessor(
        model_api_base="http://localhost:8002/v1",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        confidence_threshold=90.0
    )
    
    # Use with statement to manage client
    with processor:
        # Test case 1: Complex instruction (should decompose if confidence > 90)
        instruction1 = "Monitor pedestrian activity near the parked vehicles in urban residential areas"
        process_result1, final_instr1 = processor.process(instruction1)
        print("\n=== Test Case 1 ===")
        print("Processing result:", process_result1)
        print("Final instruction:", final_instr1)
        
        # Test case 2: Simple explicit instruction (no decomposition)
        instruction2 = "Detect trucks in the image"
        process_result2, final_instr2 = processor.process(instruction2)
        print("\n=== Test Case 2 ===")
        print("Processing result:", process_result2)
        print("Final instruction:", final_instr2)
        
        # Test case 3: Ambiguous instruction with low confidence (simulated)
        instruction3 = "Detect important objects"
        process_result3, final_instr3 = processor.process(instruction3)
        print("\n=== Test Case 3 ===")
        print("Processing result:", process_result3)
        print("Final instruction:", final_instr3)