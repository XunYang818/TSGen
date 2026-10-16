import os
from typing import Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv("TSGEN_MODEL")


# Resident profile generation prompt template
resident_profile_template = """
You are a professional character designer. Based on the following house layout and description, use creative thinking to create an ordinary person who might live here, in order to use this character profile to infer the person's possible activity trajectory in the house.

House layout:
```yaml
{layout_yaml}
```

Basic information (use this as a basis, but make reasonable adjustments):
- Occupation: {seed_occupation}
- Name: Please create an appropriate name
- Age: Please reasonably infer the age based on occupation and name
- Gender: Please generate a reasonable gender
- Family role: Main role in the family


Describe the following aspects of this ordinary resident in a realistic and practical way:
1. Basic personal information (name, age, gender, occupation, family role, etc. The name and age should be as random as possible, avoid common names)
2. Lifestyle and behavioral habits (at least 3 daily habits, such as sleep schedule, personal hobbies, work and entertainment habits, etc.)
3. Activity habits at home (frequency and manner of using each room, common activities, etc.)

Descriptions related to subsequent activity judgment should be rich, specific and reasonable, while other descriptions should be concise.
The overall character image should be three-dimensional and realistic, with both positive and negative information, like a real ordinary person.
Please ensure that the character traits match the house layout, but do not describe the room layout information, and do not include any data descriptions (such as coordinates).

Output Format Requirements:
- Start with "Character Profile" as the main header
- Use numbered sections (1. Basic Personal Information, 2. Lifestyle and Behavioral Habits, 3. Activity Habits at Home)
- Do NOT use bold formatting (** or __) anywhere in the output - use plain text only


Output should contain the above information about this person, and nothing else.
"""

# Resident profile verification prompt template
resident_verification_template = """
You are a professional character design review expert. Below are the house layout where the resident lives and the resident introduction. Please verify whether the resident introduction is reasonable, coherent and rich.

House layout:
```yaml
{layout_yaml}
```

Resident profile:
```
{resident_profile}
```

Evaluate this profile from the following aspects:
1. Whether the character image is an ordinary person, realistic, three-dimensional, and authentic.
2. Whether the character image traits and behavioral habits match the house layout (considering room layout, functions, etc.).
3. Whether the content is specific enough to provide sufficient basis for subsequent activity generation.
4. Whether there are logical contradictions or unreasonable aspects.
5. Whether the character profile contains only the character profile, without any other content, and without coordinate or data information.
6. Whether the profile uses proper structured formatting (should start with "Character Profile", have numbered sections like "1. Basic Personal Information").
7. Whether the profile contains unnecessary meta-formatting (headers like "Corrected Profile", "Summary", or redundant section labels) - if yes, this needs correction.

The room layout is fixed, do not try to correct the house layout.
Only focus on the resident profile introduction. If you find any problems, please point out the specific issues and provide correction suggestions.

Finally, please provide a clear verification result:
- If the resident introduction is reasonable and detailed, answer: "VERIFIED: Profile is reasonable and detailed"
- If there are issues that need correction, answer: "CORRECTION_NEEDED: [Detailed explanation of issues that need correction]"
"""

# Resident profile correction prompt template
resident_correction_template = """
You are a professional character designer. Correct the resident profile based on the following verification feedback.

House layout:
```yaml
{layout_yaml}
```

Current resident profile:
```
{resident_profile}
```

Verification feedback:
```
{verification_feedback}
```

Provide a complete corrected resident profile, ensuring all mentioned issues are resolved.
Note that descriptions related to subsequent activity judgment should be rich, specific and reasonable, while other descriptions should be concise.
The overall character image should be three-dimensional and realistic, with both positive and negative information, like a real ordinary person.
Please ensure that the character traits match the house layout, but do not describe the room layout information, and do not include any data descriptions (such as coordinates).

Output Format Requirements:
- Start with "Character Profile" as the main header
- Use numbered sections (1. Basic Personal Information, 2. Lifestyle and Behavioral Habits, 3. Activity Habits at Home)
- Do NOT use bold formatting (** or __) anywhere in the output - use plain text only

Output should contain the corrected character profile, and nothing else.
"""

# No longer need to extract name-age-gender


class ResidentProfileState(TypedDict):
    layout_yaml: str
    occupation: Optional[str]
    resident_profile: str
    verification_result: Optional[str]
    status: Optional[str]


def build_resident_profile_graph(model_name: str, temperature_creative: float, temperature_validation: float):
    """Build state graph for resident profile generation"""
    
    # Create LLM instances
    creative_llm = ChatOpenAI(model=model_name, temperature=temperature_creative)
    validation_llm = ChatOpenAI(model=model_name, temperature=temperature_validation)
    
    # Create chains
    resident_profile_chain = ChatPromptTemplate.from_template(resident_profile_template) | creative_llm | StrOutputParser()
    resident_verification_chain = ChatPromptTemplate.from_template(resident_verification_template) | validation_llm | StrOutputParser()
    resident_correction_chain = ChatPromptTemplate.from_template(resident_correction_template) | creative_llm | StrOutputParser()
    
    def generate_resident_profile(state: ResidentProfileState):
        """Generate resident profile"""
        layout_yaml = state["layout_yaml"]
        occupation = state.get("occupation")
        
        profile = resident_profile_chain.invoke({
            "layout_yaml": layout_yaml,
            "seed_occupation": occupation
        })
        
        return {
            "layout_yaml": layout_yaml,
            "resident_profile": profile,
            "occupation": occupation
        }

    def verify_resident_profile(state: ResidentProfileState):
        """Verify resident profile"""
        layout_yaml = state["layout_yaml"]
        resident_profile = state["resident_profile"]
        
        verification_result = resident_verification_chain.invoke({
            "layout_yaml": layout_yaml,
            "resident_profile": resident_profile
        })
        
        return {
            "layout_yaml": layout_yaml,
            "resident_profile": resident_profile,
            "verification_result": verification_result
        }

    def decide_next_step(state: ResidentProfileState):
        """Decide next step: save or correct"""
        verification_result = state["verification_result"]
        if "VERIFIED" in verification_result:
            return "save"
        else:
            return "correct"

    def correct_resident_profile(state: ResidentProfileState):
        """Correct resident profile"""
        layout_yaml = state["layout_yaml"]
        resident_profile = state["resident_profile"]
        verification_result = state["verification_result"]
        
        corrected_profile = resident_correction_chain.invoke({
            "layout_yaml": layout_yaml,
            "resident_profile": resident_profile,
            "verification_feedback": verification_result
        })
        
        return {
            "layout_yaml": layout_yaml,
            "resident_profile": corrected_profile
        }

    def save_resident_profile(state):
        """Save verified resident profile"""
        return {
            "resident_profile": state["resident_profile"]
        }
    
    workflow = StateGraph(state_schema=ResidentProfileState)
    
    # Add nodes
    workflow.add_node("generate_profile", generate_resident_profile)
    workflow.add_node("verify_profile", verify_resident_profile)
    workflow.add_node("correct_profile", correct_resident_profile)
    workflow.add_node("save_profile", save_resident_profile)
    
    # Set edges
    workflow.set_entry_point("generate_profile")
    workflow.add_edge("generate_profile", "verify_profile")
    workflow.add_conditional_edges(
        "verify_profile",
        decide_next_step,
        {
            "save": "save_profile",
            "correct": "correct_profile"
        }
    )
    workflow.add_edge("correct_profile", "verify_profile")
    workflow.add_edge("save_profile", END)
    
    return workflow.compile()


def generate_resident(layout_yaml: str, 
                      occupation: str,
                      model_name: Optional[str] = None,
                      temperature_creative: float = 0.8,
                      temperature_validation: float = 0.3) -> str:
    """
    Generate resident profile based on house layout and occupation
    Returns:
        resident profile string
    """
    model_name = model_name or MODEL_NAME or "gpt-4o-2024-11-20"

    # Create workflow
    workflow = build_resident_profile_graph(model_name, temperature_creative, temperature_validation)

    # Execute workflow
    result = workflow.invoke({
        "layout_yaml": layout_yaml,
        "occupation": occupation
    })
    
    return result["resident_profile"]
