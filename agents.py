from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI


'''
Creating agents cheatsheet:
- Think like a boss, work backwards from the goal and think about what employees you want to hire
- Define the captain of the crew who orient the other agents towards the goal
- Define which experts the captain needs to communicate with and delegate tasks to.
- Build a top down approach

Goal:
- Create a 7 day itinerary with detailed per day plans., including budget packing suggestions and safety tips.

Captain:
- Expert travel agent

Employees:
- City selection expert
- Local tour guide expert

Notes:
- IMPORTANT: Agent should be *results* driven and have a clear goal in mind
- Role is their job title
- Goals should be actionable
- Backstory should be their resume
'''

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            # Do the back story last, because the backstory should support the goal
            backstory=dedent(f"""Expert in travel planning and logistics. I have decades of experience making travel iteneraries. """),
            goal=dedent(f"""Create a 7 day travel itinerary with detailed per-day plans, 
                        include budget packing suggestions and safety tips.
                        """),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            #CUSTOMISE THIS
            llm=self.OpenAIGPT35,
        )

    def city_selection_expert(self):
        return Agent(
            role="Define agent 2 role here",
            backstory=dedent(f"""Define agent 2 backstory here"""),
            goal=dedent(f"""Define agent 2 goal here"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
    
    def local_tour_guide(self):
        return Agent(
            role="Define agent 3 role here",
            backstory=dedent(f"""Define agent 2 backstory here"""),
            goal=dedent(f"""Define agent 2 goal here"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
