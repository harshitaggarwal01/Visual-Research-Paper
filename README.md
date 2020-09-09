# IntraMIC
Research Papers are Boring!

 Project Plan:
    
    ~Objective: 

    Take a pdf of a research paper and generate a comprehensive mind map as a means of visual summarization. 

    ~Components:
    1) Extract the structured text from the pdf
        *Input: PDF*
        1) Extract text itself
        2) Identify headings and subheadings (includes table labels and captions   for visualizations)
        *Output_1: The outline of the document using text w/ tabs*
    2) Summarization of the lowest levels of content (for now: paragraph)
        *Input: Output_1*
        1) Using outline, extract paragraph content under each heading
        2) Use summarizer to summarize paragraphs under one heading as one.
            a) Set limit to how many words regardless of number of paragraphs
            b) If the introductory paragraph falls under a heading with no other subheading attached, include summary with heading node.
        3) Update heading line by appending summary 
        4) Replace content in Output_1
        *Output_2: Outline with Summarized Paragraphs*
    3) Feed Output_2 into application that will generate interactive mind map 
        *Input: Output_2 (TXT w/ Tabs)*
        1) Identifying level of association: the number of tabs detected at the beginning of a line
        2) Use Postorder Tree Traversal to create mind map's nodes
        ) Each node will hyperlink to the portion of the orginal pdf file as a webpage
        *Output_3: Interactive Mind Map = Visual Research Paper*

