# IntraMIC
## Research Papers are Boring!

 ## Project Plan:
    
   ### Objective: 

   Take a pdf of a research paper and generate a comprehensive mind map as a means of visual summarization. 

   ### Components:
   1. Extract the structured text from the pdf
        ##### Input: PDF
        1. Extract text itself
        2. Identify headings and subheadings
        ##### Output_1: The outline of the document using text w/ tabs
   2. Summarization of the lowest levels of content (for now: paragraph)
        ##### Input: Output_1
        1. Using outline, extract paragraph content under each heading
        2. Use summarizer to summarize paragraphs under one heading as one.
            1. Set limit to how many words regardless of number of paragraphs
            2. If the introductory paragraph falls under a heading with no other subheading attached, include summary with heading node.
        3. Update heading line by appending summary 
        4. Replace content in Output_1
        ##### Output_2: Outline with Summarized Paragraphs
   3. Feed Output_2 into application that will generate interactive mind map 
        ##### Input: Output_2 (TXT w/ Tabs)
        1. Identifying level of association: the number of tabs detected at the beginning of a line
        2. Use Postorder Tree Traversal to create mind map's nodes
        ##### Output_3: Interactive Mind Map = Visual Research Paper

