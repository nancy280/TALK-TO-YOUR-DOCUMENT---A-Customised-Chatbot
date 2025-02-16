what is it?
A customised chatbot for your personal use, accelerating the power of Llama2.
A chatbot that lets you talk with your own document in natural language.
One can just upload there document, it would first be preprocessed using RecursiveChracterTextSplitter and then embedding will be created to store in FAISS vector store and then it would be fed to question answering chain with Llama2 to interpret it after someone asks a question using gradio interface 

Llama2 - 

Libraries of langchain that will be used are 
1. Document loader for retrieval
2. RecursiveCharacterTextSplitter
3. HuggingFaceEmbeddings
4. FAISS(A knowledge base offered by facebook)
5. Chains

how run it?
install pip in your system by following the link(https://www.geeksforgeeks.org/how-to-install-pip-on-windows/)
now run the command (pip install time, pip install langchain and pip install gradio) in command prompt
ensure that llama-2-7b-chat.ggmlv3.q4_0.bin is present in the same directory
then execute the python file using the command(python script.py )
follow the link that appears on command prompt
