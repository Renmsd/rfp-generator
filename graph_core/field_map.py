# nodes/field_map.py

FIELD_MAP = {
    # 🧍‍♀️ User Inputs (directly entered in form)
    "Government_Agency": "input",
    "Competition_Name": "input",
    "Booklet_Number": "input",
    "Issue_Date": "input",
    "Competition_Document_Fees": "input",
    "Payment_Method": "input",
    "Name_of_Government_Agency_Representative": "input",
    "Position_of_Government_Agency_Representative": "input",
    "Phone_Number_of_Government_Agency_Representative": "input",
    "Fax_Number_of_Government_Agency_Representative": "input",
    "Email_of_Government_Agency_Representative": "input",
    "Bid_Submission_Address": "input",
    "Bid_Submission_Building": "input",
    "Bid_Submission_Floor": "input",
    "Bid_Submission_Department_Name": "input",
    "Bid_Submission_Time": "input",
    "Post_Qualification":"input",
    "Inquiry_Submission_period":"input",
    "Inquiry_Response_Period": "input",
    "Inquiry_Email":"input",
    "Initial_Guarantee_Percentage": "input",
    "Max_Penalty_Percentage":"input",
    # "Service_Delivery_Plan": "input",
    "Service_Execution_Location": "input",
    "Special_Conditions": "input",
    "Annexes":"input",

    # Project Details for context
    "Project_Type": "input",
    "Project_Duration": "input",
    "Award_Method": "input",
    "Includes_Equipment": "input",
    "Local_Content_Requirements": "input",
    # "Penalty_Deduction": "input",
    "Penalty_Execute_On_Vendor": "input",
    "Penalty_Suspend": "input",
    "Penalty_Termination": "input",



    # ⚙️ Automatically Generated (functions)
    # "Issue_Date" : "function",
    "Participation_Confirmation_Letter" :  "function",
    "Submission_of_Questions_and_Inquiries" :  "function",
    "Submission_of_Proposals" : "function",
    "Opening_of_Proposals" : "function",
    "Award_Decision_Date" : "function",
    "Commencement_of_Work" : "function",


    

    # 🧠 LLM-Generated Sections (content paragraphs)
    "Competition_Definition": "llm",
    "Text_of_Costs_of_Competition_Documents": "llm",
    "Project_Scope_of_Work": "llm",
    "Technical_Proposal_Documents": "llm",
    "Financial_Proposal_Documents": "llm",
    "Bid_Evaluation_Criteria": "llm",
    "Regulatory_Records_and_Licenses": "llm",
    "Inquiries_Section": "llm",
    "Tender_Split_Section": "llm",
    "Penalties": "llm",
    "Delay_Penalties": "llm",
    "Insurance": "llm",#هذا التامين م حطيته في الوورد 
    "Service_Delivery_Plan": "llm",
    "Bill_of_Quantities_and_Prices": "llm",
    "Materials_Specifications_Table":"llm",
    "Equipment_Specifications_Table":"llm",
    "Workers_Table":"llm",
    "Service_Execution_Method":"llm",
    "Alternative_Offers":"llm",
    "Offer_Formatting_Requirements":"llm",


    # "Alternative_Offers":"llm",
    # "Offer_Formatting_Requirements":"llm",



    # 📄 Static Defaults (seldom change)
    "Joint_Venture": "llm",
    # "Approved_Currency": "default",
}
