from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,ToolMessage
from typing_extensions import Annotated,Literal,TypedDict
import pandas as pd
from dotenv import load_dotenv
import re
import numpy as np
import plotly.express as px
import json
from cryptography.fernet import Fernet
import streamlit as st

st.title('CSV Chatbot📊')

uploaded_file = st.file_uploader("Upload a data file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.dataframe(dataframe)
    dataframe.to_csv(uploaded_file.name,index=False)


user_prompt=st.text_input('Enter your Prompt')
if st.button("Ask"):

    with open('cache_file.json','r') as f:
        cache_data=json.load(f)

    if user_prompt in cache_data.keys():
         st.header('Response:-')
         st.info(cache_data[user_prompt])
    else:
        status_message_container = st.empty()

        encrypted_key=b'gAAAAABoCkZy6AcjIj_MpvrcyzdHVpYgmG9gLTJxjtNVDtklx6pUGJs9G3asZ1uzc8JgIXNbaPtqyF05TDvYiJVtxktpZLBmGOJTUK1eiZ46fHw70Nq94eEq5797e-VePP-iFTw6sd752z8oClPpzmT1V3uneNdtPfUx1WbfTAQOaElFggdVHeMfW4nNe2xXeJtm4XVW0cfu99n2lFTs9izH7ODOkx7KgvFLc1VUENupXLg4PSNgOn_bCPTTOSIdKFGQGos98ICXjsx6wPYQpY0yoXRd1sM8OYsH0nYTo3A0Fc7YMnChuyY='

        key=b'dNxmct5HrAjsG8LsJACGnXElHLVcbzjOgsJgvdfsUck='

        fernet = Fernet(key)
        decr_api = fernet.decrypt(encrypted_key).decode()
        load_dotenv()

        # llm=ChatGroq(model='llama-3.3-70b-versatile',temperature=0.0)
        #llm=ChatCerebras(model='llama-3.3-70b',temperature=0.0)
        llm=ChatOpenAI(model='gpt-4.1-mini',temperature=0.0,api_key=decr_api)
        prompt_classifier_llm=ChatOpenAI(model='gpt-4.1-nano',temperature=0.0,api_key=decr_api)

        def python_repl(code_to_be_executed:str,repl_variables:dict):
            """
            Execute a snippet of Python code in a given namespace and return the updated namespace.

            This function allows dynamic execution of Python code strings within a
            persistent REPL-style environment. The caller provides both the code to
            run and a dictionary of variables (namespace) that will serve as the
            execution context. After running, any new or modified names in that
            context are returned for inspection or further use.

            
            Args:
                code_to_be_executed: The string containing Python code to execute.
                repl_variables: A dictionary representing the execution namespace.
                                It MUST initially contain 'df' (the DataFrame) and 'pd' (pandas).
                                The function modifies this dictionary in place and returns it."""
            
            code_to_be_executed=re.sub(r"^(\s|`)*(?i:python)?\s*", "",code_to_be_executed)
            code_to_be_executed=re.sub(r"(\s|`)*$", "",code_to_be_executed)

            print()
            print('Code to be executed\n',code_to_be_executed)

            exec(code_to_be_executed,repl_variables)
            return repl_variables

        class mainstate(TypedDict):
            csv_llm_generated_code:str
            csv_llm_response:list
            dataset_metadata:str
            user_prompt:str
            task_plan:str
            dataset_path:str
            dataset:pd.DataFrame
            repl_variables:dict
            tool_result:str
            replanner_llm_response:dict
            final_response:dict
            prompt_classifier_result:str
            data_viz_llm_generated_code:str
            
            
        def prompt_classifier(state):
            
            status_message_container.success('Classifying User Prompt...')

            prompt_classifier_system_prompt='''"You are an AI assistant designed to analyze user queries about data in the context of a text-to-SQL chatbot. Your task is to determine whether the user's query suggests that they want to see the data visualized, such as in charts, graphs, or other visual formats, rather than just receiving a table or list of data.

            To make this determination, evaluate the query based on the following criteria:

            Visualization Indicators: Does the query include words like 'show,' 'plot,' 'graph,' 'chart,' or explicitly mention a specific type of visualization (e.g., 'Sankey chart')?
            Intent for Patterns or Insights: Does the query ask for trends, patterns, distributions, comparisons, or relationships that are typically better understood through visual representation (e.g., 'trend over time,' 'market share')?
            Tabular Sufficiency: Is the query primarily asking for a list, count, or specific details that can be adequately presented in a tabular format (e.g., 'list of patients,' 'how many')?
            Based on your analysis, classify the query by responding with:

            'Yes' if the query requires data visualization.
            'No' if the query does not require data visualization and a table or list would suffice.
            Use your understanding of natural language and common data presentation practices to make this classification. Focus on the user's intent as expressed in the query's wording."

            How It Works with Your Example Questions
            To demonstrate how this template applies, here's how it would classify each of your provided questions:

            "show trend SMA patients treated at Stanford for last 3 years by month"
            Classification: Yes
            Reasoning: The word "show" combined with "trend" and a time dimension ("by month") suggests a visual representation, like a line chart, to display the pattern over time.
            "Who are the top HCPs that treat zolgensma patients and how may patients do they have that are 0-2 years of age treated with Spimraza or Evrysdi but not with spinraza"
            Classification: No
            Reasoning: This query asks "who" and "how many," focusing on specific details about healthcare providers (HCPs) and patient counts. A table listing HCPs and their respective counts would suffice.
            "There are how may that treat Ped and Adult patients and are affiliated to current IV HCOs"
            Classification: No
            Reasoning: The phrase "how many" indicates a numerical count, which can be presented as a single number or a small table. No visualization is implied.
            "List of patients < 2 years who are not treated with Zolgensma"
            Classification: No
            Reasoning: The word "list" explicitly requests a set of items (patients), which is naturally suited to a tabular format rather than a visual one.
            "Show list of Zolgensma naïve patients born after January 2023."
            Classification: No
            Reasoning: Although "show" is used, it is followed by "list," indicating a request for specific items rather than a visual pattern. A table is appropriate here.
            "show market share of sma products for last 12 months."
            Classification: Yes
            Reasoning: "Show" paired with "market share" implies a comparative view of proportions, commonly visualized as a pie chart or bar chart, rather than just a table of numbers.
            "Show Sankey charts of patient journeys across brands or specialties"
            Classification: Yes
            Reasoning: The explicit mention of "Sankey charts" clearly indicates a requirement for a specific type of data visualization.
            "There are how many patients 0-2 years of age in 2023-2024 treated by Spinraza or Evrysdi but not by Zolgensma"
            Classification: No
            Reasoning: "How many" requests a count, which can be adequately answered with a number or a small table. No visualization is suggested.

            Output Format:
            Respond by saying only 'Yes' or 'No' and don't tell anything else'''

            print("\nClassifying User Prompt...\n")
            
            prompt_classifier_response=(prompt_classifier_llm.invoke([SystemMessage(content=prompt_classifier_system_prompt),HumanMessage(content=state.get('user_prompt'))]).content)

            status_message_container.success('P')

            print('\nClassification Result:',prompt_classifier_response)

            return {'prompt_classifier_result':prompt_classifier_response}
            
        def metadata_generator(state):
            
            status_message_container.success('Creating Metadata for the Dataset...')

            df=state.get('dataset')
            df_columns=df.columns.to_list()
            df_column_dtypes={k:'string' if v=='object' else v for k,v in (df.dtypes.to_dict()).items()}
            
            sample_col_data=[]
            for col_name in df_columns:
                single_col_data={col_name:(df.loc[0:2,col_name]).to_list()}
                sample_col_data.append(single_col_data)

            column_descriptions='''
            Column Descriptions:-
            1.HCP ID: A unique identifier for the healthcare provider (HCP), such as a doctor or medical professional involved in patient care.

            2.ZOLG_PRESCRIBER: Indicates whether the healthcare provider has prescribed Zolgensma, a gene therapy used to treat SMA (values: "Yes" or "No").

            3.Zolgensma IV Target: Indicates whether the healthcare provider is targeted to prescribe Zolgensma intravenously (values: "Yes" or "No").

            4.KOL: Stands for "Key Opinion Leader." Indicates whether the healthcare provider is an influential figure in the medical community whose opinions are highly regarded (values: "Yes" or "No").

            5.PATIENT_ID: A unique identifier assigned to each patient receiving treatment.

            6.MTH: The date of the record in a year-month format (e.g., "2018_03" represents March 2018).

            7.Month: The date of the record in a month/day/year format (e.g., "3/1/18" represents March 1, 2018).

            8.DRUG_NAME: The name of the drug prescribed or administered to the patient (e.g., "SPINRAZA," "ZOLGENSMA," "EVRYSDI").

            9.AGE_GROUP: The age group of the patient, categorized into ranges such as "0 to 2" or "3 to 17."

            10.FINAL_SPEC: The final medical specialty of the healthcare provider (e.g., "CHILD NEUROLOGY," "PEDIATRIC," "RADIOLOGY").

            11.HCP_SEGMENT: A categorization or segment of the healthcare provider, possibly indicating their level or type of practice (e.g., "HIGH").

            12.HCP_NAME: The full name of the healthcare provider (e.g., "JAHANNAZ DASTGIR," "CRYSTAL PROUD").

            13.HCP_CITY: The city where the healthcare provider is located (e.g., "NEW YORK," "NORFOLK").

            14.HCP_STATE: The state where the healthcare provider is located (e.g., "NY," "VA").

            15.HCP_ZIP: The zip code of the healthcare provider's location (e.g., "10032," "23510").

            16.HCO_MDM: A unique identifier for the healthcare organization (HCO) associated with the provider (e.g., "16953373," "15046286").

            17.HCO_MDM_Name: The name of the healthcare organization (e.g., "ATLANTIC MEDICAL GROUP PEDIATRICS AT MORRISTOWN," "CHILDRENS HOSPITAL OF THE KINGS DAUGHTERS").

            18.HCO_Grouping: The type or grouping of the healthcare organization (e.g., "IV AFFILIATES," "CURRENT IV").

            19.HCO_MDM_TIER: The tier level of the healthcare organization, indicating its category or ranking (e.g., "MEDIUM").

            20.HCO_ADDR_LINE_1: The first line of the address of the healthcare organization (e.g., "435 SOUTH ST," "601 CHILDRENS LN").

            21.HCO_CITY: The city where the healthcare organization is located (e.g., "MORRISTOWN," "NORFOLK").

            22.HCO_STATE: The state where the healthcare organization is located (e.g., "NJ," "VA").

            23.HCO_POSTAL_CD_PRIM: The primary postal code (zip code) of the healthcare organization (e.g., "07960," "23507").

            24.REND_HCO_LAT: The latitude of the rendering healthcare organization, where the service is provided (e.g., "40.7952," "36.8645").

            25.REND_HCO_LONG: The longitude of the rendering healthcare organization (e.g., "-74.4873," "-76.3004").

            26.Account Setting Type: The type of account or setting for the healthcare organization (e.g., "Community").

            27.REF_NPI: The National Provider Identifier (NPI) of the referring healthcare provider (e.g., "1790750404," "1548420136"; "-" if not applicable).

            28.REF_NAME: The name of the referring healthcare provider (e.g., "JUAN GUTIERREZ," "MICHELLE SIRAK"; "-" if not applicable).

            29.REF_FINAL_SPEC: The final medical specialty of the referring healthcare provider (e.g., "PEDIATRIC," "RADIOLOGY"; "-" if not applicable).

            30.REF_HCP_ADDRESS: The address of the referring healthcare provider (e.g., "1501 N CAMPBELL AVE," "30 PROSPECT AVE BLDG 3"; "-" if not applicable).

            31.REF_HCP_CITY: The city of the referring healthcare provider (e.g., "TUCSON," "HACKENSACK"; "-" if not applicable).

            32.REF_HCP_STATE: The state of the referring healthcare provider (e.g., "AZ," "NJ"; "-" if not applicable).

            33.REF_HCP_ZIP: The zip code of the referring healthcare provider (e.g., "85724," "07601"; "-" if not applicable).

            34.REF_HCO_NPI_MDM: The NPI of the referring healthcare organization (e.g., "2190354" for "NEMOURS CHILDRENS HOSPITAL DELAWARE"; "-" if not applicable).

            35.REF_ORGANIZATION_MDM_NAME: The name of the referring healthcare organization (e.g., "NEMOURS CHILDRENS HOSPITAL DELAWARE"; "-" if not applicable).

            36.REF_HCO_ADDRESS: The address of the referring healthcare organization (e.g., "1600 ROCKLAND RD"; "-" if not applicable).

            37.REF_HCO_CITY: The city of the referring healthcare organization (e.g., "WILMINGTON"; "-" if not applicable).

            38.REF_HCO_STATE: The state of the referring healthcare organization (e.g., "DE"; "-" if not applicable).

            39.REF_HCO_ZIP: The zip code of the referring healthcare organization (e.g., "19803"; "-" if not applicable).

            40.REF_HCO_LAT: The latitude of the referring healthcare organization (e.g., "39.7994"; "-" if not applicable).

            41.REF_HCO_LONG: The longitude of the referring healthcare organization (e.g., "-75.5317"; "-" if not applicable).

            42.WITHIN/OUTSIDE HCO REFERRAL: Indicates whether the referral is within the same healthcare organization ("WITHIN"), outside it ("OUTSIDE"), or unspecified ("UNSPECIFIED").

            43.REND_TO_REF_DISTANCE(Miles): The distance in miles between the rendering and referring healthcare locations (e.g., "88.1," "0.0"; "-" if not applicable).

            44.REND_HCO_TERR_ID: The territory ID of the rendering healthcare organization (e.g., "MAGA1B").

            45.REND_HCO_TERR_NAME: The territory name of the rendering healthcare organization (e.g., "CAPITOL").

            46.REF_HCO_TERR_ID: The territory ID of the referring healthcare organization (e.g., "MAGA1B," "MAGA1A"; "-" if not applicable).

            47.REF_HCO_TERR_NAME: The territory name of the referring healthcare organization (e.g., "CAPITOL," "NEW ENGLAND"; "-" if not applicable).

            48.Zolgensma Naïve Patient: Indicates whether the patient has not previously received Zolgensma treatment (values: "Yes" or "No").

            49.L2Y_HCP Pts Potential Across Age: The potential number of patients the healthcare provider has across different age groups in the last two years (e.g., "Ped+Adults").

            50.Medicare_Flag (claim): A binary flag indicating whether the claim is related to Medicare (values: "0" for no, "1" for yes).

            51.L2Y_HCP_Medicare_Use: A binary indicator showing whether the healthcare provider has used Medicare in the last two years (values: "0" for no, "1" for yes).'''

            
            dataset_metadata=f'''Columns in Dataset:{df_columns}\n\n
                                Column Datatypes:{df_column_dtypes}\n\n
                                Sample Data in each Column:{sample_col_data}\n\n
                                Column Descriptions:\n{column_descriptions}\n\n'''
            
            return {'dataset_metadata':dataset_metadata}
                                

        def llm_node(state):
            
            status_message_container.success('Creating a Plan to answer User Prompt...')

            print('Planner LLM Execution Started...\n')

            system_prompt=system_prompt = f'''**Role:** You are a highly specialized AI Planning Agent. Your sole purpose is to meticulously analyze a user's question about a dataset and decompose it into a sequence of clear, actionable, and logically ordered tasks.

            **Target Audience for Tasks:** These tasks are **NOT** for you to execute. They are instructions for a *separate* AI agent (the "Code Execution Agent") which has access to a Python environment with the Pandas library and can execute generated code against a loaded DataFrame (typically named `df`).

            **Input You Will Receive:**
            1.  'USER_QUESTION': The natural language question asked by the user about their data.
            2.  'DATA_CONTEXT': Information about the dataset the user is asking about. This typically includes:
                *   Filename (e.g., `sales_data.csv`)
                *   Column names (e.g., `['OrderID', 'Product', 'Category', 'Price', 'Quantity', 'OrderDate', 'Region']`)
                *   Data types (e.g., `'OrderID': 'int64', 'Product': 'object', 'Price': 'float64', ...`)
                *   Sample Data in each column
                *   Description about each column

            **Your Primary Task:**
            Before generating any tasks, thoroughly review the provided dataset metadata to ensure that all required columns and data are available for answering the user's question. Then, generate a JSON formatted list of strings. Each string represents a single, discrete task that the Code Execution Agent needs to perform using Python and Pandas to ultimately gather the information required to answer the user's question which is "{state.get('user_prompt')}". If the required data is not available based on the metadata, output a list with a single string explaining why the question cannot be answered.

            **Guiding Principles for Task Creation:**
            1.  **Decomposition:** Break down the user's request into the smallest logical steps required. Complex operations (e.g., finding the top 5 products by sales in a specific region) should be multiple tasks (e.g., 1. Filter by region, 2. Group by product and sum sales, 3. Sort by sales, 4. Take top 5, 5. Extract product names/sales figures).
            2.  **Clarity & Precision:** Tasks must be unambiguous. Use precise column names provided in the DATA_CONTEXT. State exactly what calculation or manipulation is needed (e.g., "Calculate the mean of the 'Price' column", "Filter rows where 'Region' is 'West'", "Group by 'Category' and count unique 'Product'").
            3.  **Logical Sequence:** The order of tasks in the list must reflect the necessary workflow. Data loading (if implied) should often be the first step, followed by filtering, transformation, aggregation, and finally, retrieving the specific result. If any required column or data is not available based on the metadata, do not generate tasks for execution; instead, output a list with a single string explaining that the question cannot be answered due to data unavailability.
            4.  **Pandas Focus:** Assume the Code Execution Agent uses Pandas. Frame tasks in terms of DataFrame operations (loading, filtering, selecting, grouping, aggregating, sorting, merging, calculating, etc.).
            5.  **State Management (Implicit):** Assume the Code Execution Agent maintains the state of the DataFrame (`df`) between tasks. Tasks can build upon the results of previous tasks within the same list.
            6.  **Final Result:** The final task(s) should clearly state what information needs to be extracted or presented to answer the user's question. (e.g., "Return the calculated average price", "Return the list of top 5 product names", "Return the resulting filtered DataFrame count").
            7.  **Column Availability:** Since you have access to the dataset metadata, do not include tasks to check if specific columns exist. Verify the presence of required columns using the metadata. If a required column is missing, state that the question cannot be answered due to data unavailability and do not generate further tasks.

            **CRITICAL Constraints - What NOT To Do:**
            *   **DO NOT** attempt to answer the user's question yourself.
            *   **DO NOT** generate *any* Python or Pandas code. Your output is *only* the list of tasks (instructions).
            *   **DO NOT** perform calculations or data analysis. Your job is planning the steps, not executing them.
            *   **DO NOT** output anything other than the JSON list of task strings. No conversational text, no explanations outside the tasks themselves (unless the single task is to explain that the question cannot be answered).
            *   **DO NOT** make any assumptions; if you feel the question cannot be answered with the given data, mention it explicitly in the output list.
            *   **DO NOT** include tasks to check for the existence of columns or data, as you should determine this from the metadata.

            **Output Format:**
            *Strictly* output a list of strings. If the question can be answered, each string is a task for the Code Execution Agent. If the question cannot be answered due to data unavailability, output a list with a single string explaining why.

            **Example:**
            If the user's question is "What is the average price of products in the 'Electronics' category?" and the metadata shows that the 'Category' and 'Price' columns exist:

            Your Output:
            [
            "Filter the DataFrame 'df' to include only rows where the 'Category' column is equal to 'Electronics'.",
            "Calculate the mean of the 'Price' column for the filtered data.",
            "Return the calculated mean price."
            ]

            If the user's question is "What is the average age of customers?" but the metadata does not include an 'Age' column:

            Your Output:
            [
            "The question cannot be answered because the dataset does not contain an 'Age' column."
            ]

            Please find the Metadata of the dataset below:
            {state.get('dataset_metadata')}'''


            
            return {'task_plan':(llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=state.get('user_prompt'))]).content)}

        def csv_agent(state):
            status_message_container.success('Generating Code...')
            print('Task Plan:',state.get('task_plan'))
            print('\nCSV Agent Execution Started...\n')
            csva_system_prompt=f"""**Role:** You are an expert Python Code Generation Agent specializing in the Pandas library.

            **Primary Goal:** Your sole purpose is to translate a given sequence of data manipulation tasks into a single, executable block of Python code that uses the Pandas library. This code will be executed using a provided Python REPL tool.

            **Input You Will Receive:**
            1.  `TASK_PLAN`: A list of natural language strings, where each string describes a specific step to be performed on a Pandas DataFrame. This plan was generated by a planning agent.
            2.  `DATA_CONTEXT`: Metadata about the target dataset, including:
                *   Column names
                *   Column data types
                *   Sample data rows (potentially)

            **Execution Environment Context:**
            *   You **MUST** assume that a Pandas DataFrame object is already loaded and available in the execution environment under the variable name `df`.
            *   You **MUST** assume that the Pandas library is already imported and available under the alias `pd`.
            
            **Your Task:**
            1.  Carefully analyze the `TASK_PLAN` provided below.
            2.  Translate the *entire sequence* of tasks into a single, coherent block of Python code using Pandas (`pd`) functions and operating on the DataFrame `df`.
            3.  Ensure the generated code directly addresses each step in the `TASK_PLAN` in the correct order.
            4.  The code should be written such that the *final result* required by the last task(s) in the plan is implicitly available (e.g., the modified `df`, or a calculated value assigned to a variable like `result` or `final_output`). If the final task is just to display data (like `.head()` or a specific value), ensure the code performs that action (e.g., assign `final_output = df.head()`). **Prefer assigning results to a variable named `final_output` if a specific value or subset of data is requested by the final task.**
            5.  Your *only* output should be the generated Python code block as a single string.

            **CRITICAL Constraints - What NOT To Do:**
            *   **DO NOT** output *anything* other than the Python code block string. No explanations, no conversational text, no markdown formatting like ```python ... ```, no introductory phrases (e.g., "Here is the code:").
            *   **DO NOT** include the `import pandas as pd` statement in your code block. Assume `pd` is already available.
            *   **DO NOT** include code to load the CSV file (e.g., `pd.read_csv(...)`). Assume `df` is already loaded.
            *   **DO NOT** wrap your code in a function definition (`def ...:`).
            *   **DO NOT** invent columns or use columns not mentioned in the `DATA_CONTEXT`. If a task seems impossible with the given columns, generate code that attempts the task as described, relying on the execution environment to potentially raise an error if a column is truly missing. (The planner *should* have caught this, but be robust).

            **Inputs for this specific request are given below:**
            **TASK_PLAN:\n**
            {state.get('task_plan')}
            
            **DATA_CONTEXT:**
            {state.get('dataset_metadata')}

            Example OUTPUT Format which you must always follow is given below:
            "df_filtered = df[df['Sales'] > 1000].copy()
            df_filtered['OrderDate'] = pd.to_datetime(df_filtered['OrderDate'])
            df_filtered['Month'] = df_filtered['OrderDate'].dt.month
            df_filtered['Discount'] = df_filtered['Discount'].fillna(0.0)
            df_filtered['NetSales'] = df_filtered['Sales'] * (1 - df_filtered['Discount'])
            grouped_data = df_filtered.groupby(['Region', 'Month']).agg(
                TotalNetSales=('NetSales', 'sum'),
                AvgQuantity=('Quantity', 'mean')
            )
            grouped_data = grouped_data.reset_index()
            final_output = grouped_data.sort_values(by=['Region', 'TotalNetSales'], ascending=[True, False])"
            """

            csv_llm_generated_code=(llm.invoke([SystemMessage(content=csva_system_prompt)])).content

            print('CSV LLM Response:\n',csv_llm_generated_code)

            return {'csv_llm_generated_code':csv_llm_generated_code}



        def code_executor(state):
            status_message_container.success('Executing the Code...')
            print('\nCode Executor Execution Started...\n')
            repl_variables=None
            if state.get('repl_variables')==None:
                dataset=state.get('dataset')
                repl_variables={"df":dataset,"pd":pd,"np":np}
            else:
                repl_variables=state.get('repl_variables')
            
            csv_llm_generated_code=state.get('csv_llm_generated_code')
            function_result=python_repl(csv_llm_generated_code,repl_variables)
            print('\n\nFunction Result:',function_result.keys())
            function_result=function_result['final_output']
            
            return {'tool_result':function_result}

        def replanner_llm(state):
            status_message_container.success('Generating Final Response...')
            response_llm=ChatOpenAI(model='gpt-4.1-mini',temperature=0.0,api_key=decr_api)

            print('Replanner LLM Execution Started...')
            replanner_llm_system_prompt=f'''**Role:** You are an AI Final Response Generator. Your sole task is to generate a direct, natural language response to the user's original query based on the provided information.

            **Inputs You Will Receive:**

            1.  `user_prompt`: The original natural language question asked by the user.
                ```
                {state.get('user_prompt')}
                ```
            2.  `initial_task_plan`: The list of tasks initially generated by the Planner Agent. Crucially, check this first for impossibility statements.
                ```
                {state.get('task_plan')}
                ```
            3.  `tool_result`: The direct output or result obtained after executing the generated code. This is your primary source for answering the question, unless the plan indicated impossibility or an error occurred. This could be data, a confirmation message, or an error message.
                ```
                {state.get('tool_result')}
                ```

            **Your Task and Logic:**

            1.  **Check for Explicit Impossibility in Plan:**
                *   Examine the `initial_task_plan`. Does it contain explicit phrases indicating the question cannot be answered due to missing data (e.g., "cannot be answered because", "data not available", "does not contain", "missing column", "lacks the required information")?
                *   **If YES:** Your *entire output* must be the explanation provided *within the `initial_task_plan`*. Extract that explanation directly. Do not add any other text.

            2.  **Check for Execution Error in Result:**
                *   **If NO to step 1:** Examine the `tool_result`. Does it clearly indicate an error occurred during execution (e.g., starts with "Error:", contains "Traceback", `KeyError`, `TypeError`)?
                *   **If YES:** Your *entire output* must be a concise message stating that the analysis could not be completed due to an error during execution. You can optionally mention the type of error briefly if available in `tool_result` (e.g., "Analysis failed due to a KeyError during data processing."). Do not output the full traceback.

            3.  **Generate Answer from Result:**
                *   **If NO to step 1 AND NO to step 2:** Synthesize a clear, natural language answer to the `user_prompt` based **SOLELY** on the information present in the `tool_result`.
                    *   Directly address the `user_prompt`.
                    *   If `tool_result` contains the specific data (number, list, text), incorporate it naturally into the answer.
                    *   If `tool_result` is complex (like a table snippet or dictionary), summarize the key information relevant to the `user_prompt`.
                    *   If `tool_result` just confirms success without specific data ("Code executed successfully..."), state that the requested action was performed.
                    *   If `tool_result` contains information, but it doesn't fully answer the `user_prompt`, answer what you can based *only* on the `tool_result` and state what is available (e.g., "The analysis provided the following data: [summary of tool_result], but does not contain the specific average value requested.").
                    *   **CRITICAL: DO NOT** invent data, make assumptions beyond `tool_result`, or refer back to the code or plan (unless explaining impossibility as per step 1). Your output is the final user-facing answer.

            **Output Format:**

            *   **CRITICAL:** Your *entire* output **MUST** be **ONLY** the final natural language response string.
            *   **DO NOT** include *any* preamble (like "Here is the response:"), labels, apologies, JSON formatting, or markdown (like ```). Just the answer or explanation itself.

            **Example Output (Scenario 1 - Data Unavailable from Plan):**
            The question cannot be answered because the dataset does not contain an 'Age' column.

            **Example Output (Scenario 2 - Execution Error):**
            The analysis could not be completed due to an error during data processing.

            **Example Output (Scenario 3 - Success):**
            The average price for products in the 'Electronics' category is $155.75.

            **Example Output (Scenario 3 - Success, Partial Answer):**
            The analysis identified the top 5 products: [Product A, Product B, Product C, Product D, Product E]. The overall average price across all categories was not calculated in this result.
            '''

            print()
            
            replanner_llm_response=(response_llm.invoke([SystemMessage(content=replanner_llm_system_prompt)])).content
        
            user_prompt=state.get('user_prompt')

            with open('cache_file.json','w') as f:
                json.dump({user_prompt:replanner_llm_response},f)

            return {'final_response':{'final_response':replanner_llm_response}}

        def dataviz_planner_node(state):
                status_message_container.success('Creating a Plan to answer User Prompt...')
                print('\nPlanner Execution started........\n')
                dataviz_planner_llm_system_prompt=f'''**Role:** You are a highly specialized AI Planning Agent. Your sole purpose is to meticulously analyze a user's question about a dataset (which is guaranteed to be suitable for plotting) and decompose it into a sequence of clear, actionable, and logically ordered tasks *to create the required plot*.

                **Target Audience for Tasks:** These tasks are **NOT** for you to execute. They are instructions for a *separate* AI agent (the "Code Execution Agent") which has access to a Python environment with Pandas and plotting libraries like Plotly, and can execute generated code against a loaded DataFrame (typically named `df`).

                **Input You Will Receive:**
                1.  'USER_QUESTION': The natural language question asked by the user about their data. **You can assume this question inherently asks for a visual representation.**
                2.  'DATA_CONTEXT': Information about the dataset the user is asking about. This **includes definitive information** on:
                    *   Filename (e.g., `library_data.csv`)
                    *   Column names (e.g., `['Book_ID', 'Title', 'Author', 'Genre', 'Status', 'Condition', 'Checkout_Date', 'Times_Borrowed']`)
                    *   Data types (e.g., `'Book_ID': 'object', 'Times_Borrowed': 'int64', ...`)
                    *   Column Descriptions and potentially sample data.

                **Your Primary Task:**
                Your primary task is to verify, using **only** the provided DATA_CONTEXT, that all columns mentioned or implied by the user's question '{state.get('user_prompt')}' actually exist in the dataset.
                *   If **YES** (all necessary columns are confirmed present in the DATA_CONTEXT), generate a JSON formatted list of strings. Each string represents a single, discrete task that the Code Execution Agent needs to perform using Python (Pandas, Plotly) to ultimately *generate the required plot*.
                *   If **NO** (**any required column is missing from the DATA_CONTEXT**), your output must be *only* the exact string: `Cannot be answered through a plot`

                **Guiding Principles for Task Creation (if columns are available):**
                1.  **Assume Column Availability (Post-Verification):** Since you have already verified column existence based on the DATA_CONTEXT (as per the Primary Task description), all subsequent tasks should assume these columns are available and directly use their names from the DATA_CONTEXT.
                2.  **Decomposition:** Break down the process into the smallest logical steps. This includes data preparation (filtering, grouping, sorting, aggregation) *specifically needed for the plot*, followed by plot creation steps.
                3.  **Clarity & Precision:** Tasks must be unambiguous. Use precise column names *known to exist* from the DATA_CONTEXT. State exactly what calculation, manipulation, or plot configuration is needed (e.g., "Calculate the sum of 'Times_Borrowed' grouped by 'Genre'", "Filter rows where 'Status' is 'Available'", "Set 'Genre' as the x-axis", "Set the calculated sum as the y-axis").
                4.  **Logical Sequence:** The order of tasks must reflect the workflow: Data selection/preparation -> Plot definition -> Plot finalization/storage.
                5.  **Data Prep & Plot Focus:** Frame tasks in terms of DataFrame operations needed to prepare data for plotting *and* defining the plot itself (selecting plot type like bar, line, scatter; specifying x/y axes, titles, labels, colors if necessary). Assume the Code Execution Agent uses Plotly.
                6.  **State Management (Implicit):** Assume the Code Execution Agent maintains the state of the DataFrame (`df`) and intermediate data structures between tasks.
                7.  **Final Result:** The final task(s) should clearly instruct the Code Execution Agent to create the plot object (e.g., using `plotly.express`) and store it in a designated variable named exactly `final_output_fig` for later use/display.

                **CRITICAL Constraints - What NOT To Do:**
                *   **DO NOT** attempt to answer the user's question yourself or generate the plot image/data.
                *   **DO NOT** generate *any* Python code. Your output is *only* the list of tasks *or* the specific "Cannot..." string.
                *   **DO NOT** perform calculations or data analysis. Your job is planning the steps, not executing them.
                *   **DO NOT** output anything other than the JSON list of task strings *OR* the exact string `Cannot be answered through a plot`. No conversational text, no explanations outside the tasks.
                *   **DO NOT** make assumptions if data is missing; default to `Cannot be answered through a plot` **based on your initial check of the DATA_CONTEXT.**
                *   **ABSOLUTELY DO NOT generate tasks to check for the existence of columns (e.g., "Check if 'Genre' column exists") or to check for the presence of specific values within columns (e.g., "Verify 'Available' exists in the 'Status' column"). You have already confirmed column existence using the DATA_CONTEXT before deciding to generate tasks.** Your tasks should *directly use* the columns assumed to be present.

                **Output Format:**
                *   **If required columns are present:** Strictly output a JSON formatted list of strings. `["Task 1", "Task 2", ...]`
                *   **If any required column is missing (based on DATA_CONTEXT check):** Strictly output *only* the following string: `Cannot be answered through a plot`

                **Example (Columns Available):**
                If the user's question is "Show the number of books per genre?" and DATA_CONTEXT confirms 'Genre' and 'Book_ID' exist:

                Your Output (Example Only):
                [
                "Group the DataFrame 'df' by the 'Genre' column.",
                "Calculate the count of entries using 'Book_ID' for each genre.",
                "Reset the index of the grouped data to make 'Genre' and the count regular columns.",
                "Rename the count column to 'Number_of_Books'.",
                "Create a bar plot using Plotly Express.",
                "Set the 'Genre' column as the x-axis.",
                "Set the 'Number_of_Books' column as the y-axis.",
                "Set the plot title to 'Number of Books per Genre'.",
                "Store the resulting Plotly figure object in the variable 'final_output_fig'."
                ]

                **Example (Missing Column in DATA_CONTEXT):**
                If the user's question is "Show book condition vs times borrowed?" but DATA_CONTEXT lacks a 'Condition' column:

                Your Output (Example Only):
                Cannot be answered through a plot

                Please find the Metadata of the dataset below
                {state.get('dataset_metadata')}'''

                plan=(llm.invoke([SystemMessage(content=dataviz_planner_llm_system_prompt)])).content
                print('Task Plan:',plan)

                return {'task_plan':plan}

        def dataviz_code_gen_node(state):
                status_message_container.success('Generating Code...')
                print('\n\nCode Generation started......\n')

                code_gen_llm_system_prompt_template=f"""**Role:** You are an expert Python Code Generation Agent specializing in using the Pandas library for data manipulation and the Plotly Express library for data visualization.

                **Primary Goal:** Your sole purpose is to translate a given sequence of data preparation and plotting tasks into a single, executable block of Python code. This code will use Pandas (`pd`) for data handling and Plotly Express (`px`) to generate a plot object based on the plan. This code will be executed using a provided Python REPL tool.

                **Input You Will Receive:**
                1.  `TASK_PLAN`: A list of natural language strings, where each string describes a specific step (data manipulation or plot configuration) to be performed. This plan was generated by a planning agent.
                2.  `DATA_CONTEXT`: Metadata about the target dataset, including:
                    *   Filename
                    *   Column names
                    *   Column data types
                    *   Sample data rows (potentially)

                **Execution Environment Context:**
                *   You **MUST** assume that a Pandas DataFrame object is already loaded and available in the execution environment under the variable name `df`.
                *   You **MUST** assume that the Pandas library is already imported and available under the alias `pd`.
                *   You **MUST** assume that the Plotly Express library is already imported and available under the alias `px`.

                **Your Task:**
                1.  Carefully analyze the `TASK_PLAN` provided below.
                2.  Translate the *entire sequence* of tasks into a single, coherent block of Python code using Pandas (`pd`) for any necessary data preparation and Plotly Express (`px`) for generating the plot. The code must operate on the DataFrame `df`.
                3.  Ensure the generated code directly addresses each step in the `TASK_PLAN` in the correct order. Data preparation steps should precede plotting steps.
                4.  The final step in your generated code **MUST** be the creation of a Plotly figure object using a `px` function (e.g., `px.bar`, `px.line`, `px.scatter`).
                5.  This final Plotly figure object **MUST** be assigned to a variable named exactly `final_output_fig`.
                6.  Your *only* output should be the generated Python code block as a single string.
                7.  If in the Task plan if anything is mentioned that graphs cannot be created for the user prompt then politely respond to the user that it is not possible to create a graph and do not tell anything else.

                **CRITICAL Constraints - What NOT To Do:**
                *   **DO NOT** output *anything* other than the Python code block string. No explanations, no conversational text, no markdown formatting like ```python ... ```, no introductory phrases (e.g., "Here is the code:").
                *   **DO NOT** include `import pandas as pd` in your code block. Assume `pd` is already available.
                *   **DO NOT** include `import plotly.express as px` in your code block. Assume `px` is already available.
                *   **DO NOT** include code to load the CSV file (e.g., `pd.read_csv(...)`). Assume `df` is already loaded.
                *   **DO NOT** wrap your code in a function definition (`def ...:`).
                *   **DO NOT** include any code to display the plot (e.g., `final_output_fig.show()`). Only generate the code that *creates* the figure object and assigns it to `final_output_fig`.
                *   **DO NOT** invent columns or use columns not mentioned in the `DATA_CONTEXT`. If a task seems impossible with the given columns, generate code that attempts the task as described, relying on the execution environment to potentially raise an error.

                **Inputs for this specific request are given below:**
                **TASK_PLAN:**
                {state.get('task_plan')}

                **DATA_CONTEXT:**
                {state.get('dataset_metadata')}

                **Example OUTPUT Format which you must always follow is given below (Example assumes TASK_PLAN was for plotting top 5 borrowed books):**
                top_books = df.sort_values(by='Times_Borrowed', ascending=False).head(5)
                final_output_fig = px.bar(top_books, x='Title', y='Times_Borrowed', title='Top 5 Most Borrowed Books', labels={{'Times_Borrowed': 'Number of Times Borrowed', 'Title': 'Book Title'}}, text='Times_Borrowed')
                final_output_fig.update_traces(textposition='outside')
                final_output_fig.update_layout(xaxis_tickangle=-30)
                """
                llm_generated_code=(llm.invoke([SystemMessage(content=code_gen_llm_system_prompt_template)])).content
                print('LLM Generated Code:\n',llm_generated_code)
                return {'data_viz_llm_generated_code':llm_generated_code} 

        def dataviz_code_exe_node(state):
                status_message_container.success('Executing the Code and Creating Visualization...')
                print('\n\nCode Execution Started......\n')
                code_to_be_executed=state.get('data_viz_llm_generated_code')
                df=state.get('dataset')
                repl_variables={"df":df,"pd":pd,"px":px}
                llm_code_execution_result=python_repl(code_to_be_executed,repl_variables)
                print('Repl_Variables:',llm_code_execution_result.keys())
                final_graph_response=llm_code_execution_result['final_output_fig']
                status_message_container.markdown("")
                return {'final_response':{'final_response':final_graph_response}}

        def prompt_classifier_conditional_node(state):
            prompt_classifier_res=state.get('prompt_classifier_result')
            if prompt_classifier_res.lower()=='yes':
                return 'Data Viz Planner'
            else:
                return 'Planner LLM'

        builder=StateGraph(mainstate)
        builder.add_node('Prompt Classifier',prompt_classifier)
        builder.add_node('Metadata Generator',metadata_generator)
        builder.add_node('Planner LLM',llm_node)
        builder.add_node('Code Gen LLM',csv_agent)
        builder.add_node('Code Executor',code_executor)
        builder.add_node('Response Generator',replanner_llm)
        builder.add_node('Data Viz Planner',dataviz_planner_node)
        builder.add_node('Data Viz Code Generator',dataviz_code_gen_node)
        builder.add_node('Data Viz Code Executor',dataviz_code_exe_node)

        builder.add_edge(START,'Prompt Classifier')
        builder.add_edge('Prompt Classifier','Metadata Generator')
        builder.add_conditional_edges('Metadata Generator',prompt_classifier_conditional_node,['Planner LLM','Data Viz Planner'])
        builder.add_edge('Planner LLM','Code Gen LLM')
        builder.add_edge('Code Gen LLM','Code Executor')
        builder.add_edge('Code Executor','Response Generator')
        builder.add_edge('Response Generator',END)

        builder.add_edge('Data Viz Planner','Data Viz Code Generator')
        builder.add_edge('Data Viz Code Generator','Data Viz Code Executor')
        builder.add_edge('Data Viz Code Executor',END)

        with st.spinner(text="Analyzing your data"):
            graph=builder.compile()

            graph_result=graph.invoke({'user_prompt':user_prompt,'dataset':dataframe})

            if isinstance(graph_result['final_response']['final_response'],str):
                st.header("Response:-")
                st.info(graph_result['final_response']['final_response'])

            else:
                st.header("Response:-")
                st.plotly_chart(graph_result['final_response']['final_response'])

