from smolagents import CodeAgent,DuckDuckGoSearchTool, InferenceClientModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from dotenv import load_dotenv
from tools.final_answer import FinalAnswerTool
# Find Clayton Kershaw's player id
from pybaseball import  playerid_lookup
from pybaseball import  statcast_pitcher
import pandas as pd

from Gradio_UI import GradioUI

load_dotenv()

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def pitching_tool(start_date:str, end_date:str, pitcher_first_name: str, pitcher_last_name:str)-> pd.DataFrame: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that gets the average velocity of each pitch thrown by the requested pitcher
    Args:
        start_date: the start date to pull statcast data from, formatted YYYY-MM-DD
        end_date: the end date to pull statcast data from, formatted YYYY-MM-DD
        pitcher_first_name: the pitcher's first name
        pitcher_last_name: the pitcher's last name
    """
    lookup = playerid_lookup(pitcher_last_name, pitcher_first_name)
    player_id = int(lookup.iloc[0]["key_mlbam"])

    # His MLBAM ID is 477132, so we feed that as the player_id argument to the following function
    pitcher_stats = statcast_pitcher(start_date, end_date, player_id)
    average_velo = pitcher_stats.groupby("pitch_type").release_speed.agg("mean")

    return average_velo

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = InferenceClientModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
#image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompt.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, pitching_tool, get_current_time_in_timezone], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()