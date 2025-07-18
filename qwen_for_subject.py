from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

class Qwen2VL:
    def __init__(self, vl_model_path, llm_model_path, filter_prompt=True, is_detailed=False,):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            vl_model_path, torch_dtype="auto", device_map="cuda:0"
        )
        self.processor = AutoProcessor.from_pretrained(vl_model_path)
        
        self.detailed_prompt = """
        Describe this {} briefly and precisely in max 20 words, focusing on its overall appearance and key distinguishing features.
        """
        
        ## for style transfer
        # self.detailed_prompt = """
        # Describe this {} style briefly and precisely in max 20 words, focusing on its aesthetic qualities, visual elements, and distinctive artistic characteristics.
        # """
        if is_detailed:
            self.detailed_prompt = """
                [Task Description]
                As an experienced image analyst, your task is to provide a detailed description of the main features and characteristics of the given {} in this image according to the following criteria.

                [Feature Analysis Criteria]
                Analyze and describe the following visual elements:
                1. Shape
                - Main body outline
                - Overall structure
                - Proportions and composition
                - Spatial organization

                2. Color
                - Color palette and schemes
                - Saturation levels
                - Brightness/contrast
                - Color distribution patterns

                3. Texture
                - Surface qualities
                - Detail clarity
                - Visual patterns
                - Material appearance

                4. Subject-Specific Features
                - If human/animal: facial features, expressions, poses
                - If object: distinctive characteristics, condition
                - If landscape: environmental elements, atmosphere

                [Description Quality Levels]
                Your description should aim for the highest level of detail:
                Level 1: Basic identification of main elements
                Level 2: Description of obvious features
                Level 3: Detailed analysis of multiple characteristics
                Level 4: Comprehensive analysis with subtle details

                [Output Format]
                Please provide your analysis in the following structure:

                Main Subject: [Brief identifier]
                Primary Features:
                - Shape: [Description]
                - Color: [Description]
                - Texture: [Description]
                - Subject-Specific Details: [Description]
                Overall Composition: [Brief summary]
            """
        
        if filter_prompt:
            model_id_qwen2 = llm_model_path
            self.qwen2 = AutoModelForCausalLM.from_pretrained(
                model_id_qwen2,
                torch_dtype="auto",
                device_map="cuda:0"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id_qwen2)
            
            self.filter_prompt = "Please extract only the physical characteristics and features of the main {} from this description, removing any information about actions, environment, background, other subjects, or surrounding elements. Return only the extracted description without any additional commentary. The description is: \'{}\'"
            # self.filter_prompt = """Please extract only the stylistic and artistic characteristics of the {} from this description, removing any information about physical objects, specific subjects, narrative elements, or factual content. Focus solely on the aesthetic qualities, visual techniques, artistic movements, and distinctive style elements. Return only the extracted style description without any additional commentary. The description is: \'{}\'"""
            
    def get_filtered_description(self, origin_prompt, subject_name='subject'):
        prompt = self.filter_prompt.format(subject_name, origin_prompt)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.qwen2.generate(
            **model_inputs,
            max_new_tokens=512,
            # temperature=0.7,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
        
    def get_description(self, image_path, subject_name='subject'):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": self.detailed_prompt.format(subject_name, subject_name)},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text
