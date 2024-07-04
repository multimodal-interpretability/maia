// Function to populate the dropdown menu with available models
function populateModelDropdown(models) {
    const modelDropdown = document.getElementById('modelDropdown');
    modelDropdown.innerHTML = ''; // Clear existing options
    generateTaskDescription(models[0])
    // const taskDescription = document.getElementById("taskDescription");
    // taskDescription.innerHTML = ''

    models.forEach((model, index) => {
        let option = new Option(model.name, index);
        modelDropdown.add(option);
    });

    modelDropdown.addEventListener('change', (event) => {
        taskDescription.innerHTML = ''
        generateHTMLFromJSON(models[event.target.value]);
        generateTaskDescription(models[event.target.value])
    });
    // document.getElementById('modelDropdown').textContent = model.name;

}

function generateTaskDescription(model){
    const taskDescription = document.getElementById("taskDescription");
    const description = document.createElement("div");
    description.className = "task";
    if(model.name.includes('Bias')){
        description.innerHTML = `In this task, MAIA is prompted to investigate biases in the outputs of a classifier (ResNet-152) trained on ImageNet classification.`;
    } else if(model.name.includes('Spurious')){
        description.innerHTML = `In this task, MAIA is prompted to identify which neurons are sensitive to spurious features in a ResNet-18 classifier trained to identify dog breeds in different environments.`;
    } else {
        description.innerHTML = `In this task, MAIA is prompted to describe neuron-level behavior inside several pre-trained vision models.`
        
    }
    taskDescription.appendChild(description);
}

function createCodeBlock(codeText) {
    const codeBlock = document.createElement("pre");
    codeBlock.className = "code-block";
    
    // This is where you would add spans with classes for syntax highlighting
    // For a real implementation, you'd want to parse the codeText and wrap syntax parts in spans
    // Here's a simplified static example
    const highlightedCode = codeText
        // .replace(/(def)/g, '<span class="string">$&</span>') // Strings
        // .replace(/(def|class)\b/g, '<span class="keyword">$&</span>') // Keywords
        .replace(/(['"])(?:(?=(\\?))\2.)*?\1/g, '<span class="string">$&</span>') // Strings
        .replace(/#.*/g, '<span class="comment">$&</span>'); // Comments

    codeBlock.innerHTML = highlightedCode;

    return codeBlock;
}

function createBlock(content) {
    const block = document.createElement("pre");
    block.className = "block";
    block.innerHTML = content;
    return block;
}

function createImageTextLine(imagePath, textContent) {
    // Create container div
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';

    // Create the image element
    const image = new Image();
    image.src = imagePath; // Set the source of your image
    image.alt = 'Descriptive text'; // Set the alt text
    image.style.maxWidth = '50px'; // Or whatever size you prefer
    image.style.height = 'auto'; // Maintain aspect ratio
    image.style.marginRight = '10px'; // Optional space between the image and text
    image.style.marginLeft = '10px'; // Optional space between the image and text
    container.appendChild(image); // Append the image to the container

    // Create the text div
    const textDiv = document.createElement('div');
    const textNode = document.createTextNode(textContent); // Create text node
    textDiv.style.fontSize = '1.2em'; // Set the font size
    textDiv.style.fontFamily = 'Lato';
    textDiv.style.fontWeight = '800'; /* For a bold look in titles */
    textDiv.style.fontWeight = 'True'
    // textDiv.style.verticalAlign= 'top';
    textDiv.appendChild(textNode); // Append the text node to the text div
    container.appendChild(textDiv); // Append the text div to the container

    return container; // Return the complete container
}


function addExperimentDetails(experiment, userQuery1, userQuery2) {
    const element = document.createElement("pre");

    const imageTextLineElement = createImageTextLine('./user.png', 'User Query');
    element.appendChild(imageTextLineElement);
    const textDiv = document.createElement("div");
    textDiv.className = "experiment-text";
    textDiv.innerHTML = `${userQuery1}`
    const codeBlock = document.createElement("pre");
    codeBlock.className = "code-block";
    const formattedInstructions = userQuery2.replace(/\n/g, '<br>');
    const collapsibleBlock = createCollapsibleBlock("Read more...",formattedInstructions, "none")
    textDiv.appendChild(collapsibleBlock);
    element.appendChild(textDiv);

    // element.className = "experiment-details";
    // const detailsDiv = document.getElementById("experimentDetails");
    experiment.forEach(item => {
        if (item.role === "assistant") {
            const imageTextLineElement = createImageTextLine('./maia_robot.png', 'MAIA');
            element.appendChild(imageTextLineElement);
            item.content.forEach(content => {
                if (content.type === "text") {
                    if (content.text.startsWith("[CODE]:")) {
                        const codeBlock = document.createElement("pre");
                        codeBlock.className = "code-block";
                        codeBlock.textContent = content.text
                        .replace(/\[CODE\]:/g, '')                 // Removes "[CODE]:"
                        .replace(/```python/g, '')                 // Removes "```python"
                        .replace(/def execute_command\(system, tools\):/g, '')  // Removes "def execute_command(system, tools):"
                        .replace(/```/g, '')                       // Removes "```"
                        .replace(/^\s*[\r\n]/gm, '');              // Removes empty lines
                        // content.text.remove("[CODE]: \n").remove("```python\n").remove("def execute_command(system, tools):\n").remove("```", "").remove('\n\n');
                            // codeBlock.textContent = content.text.replace("[CODE]: \n```python\ndef execute_command(system, tools):\n", "").replace("```", "");
                        // codeBlock.textContent = content.text;
                        const collapsibleCodeBlock = createCollapsibleCodeBlock(codeBlock.textContent);
                        element.appendChild(collapsibleCodeBlock);
                    } else {
                        const textDiv = document.createElement("div");
                        // textDiv.textContent = content.text.replace("[HYPOTHESIS LIST]:", "Hypothesis list:");
                        textDiv.className = "experiment-text";
                        // textDiv.innerHTML = content.text.replace(/\n/g, '<br>').replace("[HYPOTHESIS LIST]:", "<b>Hypothesis list:</b>").replace("[DESCRIPTION]:", "<b>Description:</b>").replace("[LABEL]:", "<b>Label:</b>").replace("image:", "");;
                        textDiv.innerHTML = content.text
                            .replace(/\n/g, '<br>')
                            .replace(/\[HYPOTHESIS LIST\]:/g, "<u>Hypothesis list</u>:")
                            .replace(/\[DESCRIPTION\]:/g, "<u>Description</u>:")
                            .replace(/\[LABEL\]:/g, "<u>Label</u>:")
                            .replace(/\[BIAS\]:/g, "<u>Bias</u>:")
                            .replace(/\[REASONING\]:/g, "<u>Reasoning</u>:")
                            .replace(/image:/g, "");
    
                        element.appendChild(textDiv);
                    }
                } else if (content.type === "image_url") {
                    const image = new Image();
                    image.src = content.image_url;
                    image.className = "experiment-image";
                    element.appendChild(image);
                }
            });
        }
        else if (item.role === "user"){
            const imageTextLineElement = createImageTextLine('./gears.png', 'Experiment Execution');
            element.appendChild(imageTextLineElement);
            // image.src = "https://cdn3.iconfinder.com/data/icons/automobile-icons/439/Gearwheel-512.png";
            
            const table = document.createElement('table');
            table.className = 'content-table'; // Assuming you have CSS for this class

            let row = document.createElement('tr'); // Initialize the first row
            let cellCount = 0; // To keep track of the number of cells in the current row

            if (item.content.length===1){
                const content = item.content[0];
                const textDiv = document.createElement("div");
                // textDiv.textContent = content.text.replace("[HYPOTHESIS LIST]:", "Hypothesis list:");
                textDiv.className = "exe-text";
                // textDiv.innerHTML = content.text.replace(/\n/g, '<br>').replace("[HYPOTHESIS LIST]:", "<b>Hypothesis list:</b>").replace("[DESCRIPTION]:", "<b>Description:</b>").replace("[LABEL]:", "<b>Label:</b>").replace("image:", "");;
                textDiv.innerHTML = content.text
                        .split('\n')  // Split the text into an array of lines
                        .filter(line => !line.startsWith("Max activation is smaller than"))  // Filter out lines that start with the specific phrase
                        .join('<br>')  // Join the remaining lines back together, using <br> for line breaks in HTML                   
                        .replace(/\n/g, '<br>')
                        .replace(/activation:/g, '<br>activation:')
                        .replace(/\[HYPOTHESIS LIST\]:/g, "<u>Hypothesis list</u>:")
                        .replace(/\[DESCRIPTION\]:/g, "<u>Description</u>:")
                        .replace(/\[LABEL\]:/g, "<u>Label</u>:")
                        .replace(/image:/g, "");
                element.appendChild(textDiv);
            } else {
                for (let i = 0; i < item.content.length - 1; i++) {
                    const content = item.content[i];
                    const nextContent = item.content[i + 1];

                    // Check if the current content is text and the next content is an image
                    if (content.type === "text" && nextContent && nextContent.type === "image_url") {
                        const cell = document.createElement('td'); // Create a new cell for this text-image pair

                        // Create and append the text div
                        const textDiv = document.createElement("div");
                        textDiv.className = "exe-text";
                        textDiv.innerHTML = content.text
                            .split('\n')  // Split the text into an array of lines
                            .filter(line => !line.startsWith("Max activation is smaller than"))  // Filter out lines that start with the specific phrase
                            .join('<br>')  // Join the remaining lines back together, using <br> for line breaks in HTML                   
                            .replace(/\n/g, '<br>')
                            .replace(/activation:/g, '<br>activation:')
                            .replace(/\[HYPOTHESIS LIST\]:/g, "<u>Hypothesis list</u>:")
                            .replace(/\[DESCRIPTION\]:/g, "<u>Description</u>:")
                            .replace(/\[LABEL\]:/g, "<u>Label</u>:")
                            .replace(/image:/g, "");
                        cell.appendChild(textDiv);

                        // Create and append the image
                        const image = new Image();
                        image.src = nextContent.image_url;
                        image.className = "experiment-image";
                        cell.appendChild(image);

                        // Add the cell to the current row and increment the cell count
                        row.appendChild(cell);
                        cellCount++;

                        // Move to the next content after the image
                        i++;

                        // If the row has 5 cells or this is the last content pair, append the row to the table and start a new row
                        if (cellCount === 4 || i >= item.content.length - 2) {
                            table.appendChild(row);
                            row = document.createElement('tr'); // Start a new row
                            cellCount = 0; // Reset cell count for the new row
                        }
                    }
                    table.appendChild(row);
                    element.appendChild(table); 

                    if(content.type === "text"  && nextContent.type != "image_url") {
                        // table.appendChild(row);
                        // Append the table to the element (make sure 'element' is defined and points to a valid container)
                        // element.appendChild(table);

                        const textDiv = document.createElement("div");
                        // textDiv.textContent = content.text.replace("[HYPOTHESIS LIST]:", "Hypothesis list:");
                        textDiv.className = "exe-text";
                        // textDiv.innerHTML = content.text.replace(/\n/g, '<br>').replace("[HYPOTHESIS LIST]:", "<b>Hypothesis list:</b>").replace("[DESCRIPTION]:", "<b>Description:</b>").replace("[LABEL]:", "<b>Label:</b>").replace("image:", "");;
                        textDiv.innerHTML = content.text
                                .split('\n')  // Split the text into an array of lines
                                .filter(line => !line.startsWith("Max activation is smaller than"))  // Filter out lines that start with the specific phrase
                                .join('<br>')  // Join the remaining lines back together, using <br> for line breaks in HTML                   
                                .replace(/\n/g, '<br>')
                                .replace(/activation:/g, '<br>activation:')
                                .replace(/\[HYPOTHESIS LIST\]:/g, "<u>Hypothesis list</u>:")
                                .replace(/\[DESCRIPTION\]:/g, "<u>Description</u>:")
                                .replace(/\[LABEL\]:/g, "<u>Label</u>:")
                                .replace(/image:/g, "");
                        element.appendChild(textDiv);

                        if(i === item.content.length - 2){
                            const textDivNext = document.createElement("div");
                            // textDiv.textContent = content.text.replace("[HYPOTHESIS LIST]:", "Hypothesis list:");
                            textDivNext.className = "exe-text";
                            // textDiv.innerHTML = content.text.replace(/\n/g, '<br>').replace("[HYPOTHESIS LIST]:", "<b>Hypothesis list:</b>").replace("[DESCRIPTION]:", "<b>Description:</b>").replace("[LABEL]:", "<b>Label:</b>").replace("image:", "");;
                            textDiv.innerHTML =  nextContent.text
                                    .split('\n')  // Split the text into an array of lines
                                    .filter(line => !line.startsWith("Max activation is smaller than"))  // Filter out lines that start with the specific phrase
                                    .join('<br>')  // Join the remaining lines back together, using <br> for line breaks in HTML                   
                                    .replace(/\n/g, '<br>')
                                    .replace(/activation:/g, '<br>activation:')
                                    .replace(/\[HYPOTHESIS LIST\]:/g, "<u>Hypothesis list</u>:")
                                    .replace(/\[DESCRIPTION\]:/g, "<u>Description</u>:")
                                    .replace(/\[LABEL\]:/g, "<u>Label</u>:")
                                    .replace(/image:/g, "");
                            element.appendChild(textDivNext);
                        }
                    }
                }
            }   
        }
    });
    return element
}



function generateHTMLFromJSON(model) {
    const networkContainer = document.getElementById("networkContainer");
    networkContainer.innerHTML = ''; // Clear existing content

    model.layers.forEach((layer, layerIndex) => {
        const layerContainer = document.createElement("div");
        layerContainer.className = "layer-container";
        networkContainer.appendChild(layerContainer);

        const layerTitle = document.createElement("div");
        layerTitle.className = "layer-title";
        // layerTitle.textContent = `Layer ${layerIndex + 1}`;
        layerTitle.textContent = layer.id;
        layerContainer.appendChild(layerTitle);

        const layerDiv = document.createElement("div");
        layerDiv.className = "layer";
        layerContainer.appendChild(layerDiv);
        
        layer.units.forEach((unit, unitIndex) => {
            const unitDiv = document.createElement("div");
            unitDiv.className = "unit";
            unitDiv.setAttribute('data-label', unit.label);
            unitDiv.addEventListener('click', () => showExperimentDetails(unit.experiment, model.name, layer.id, unit.id, unit.label, unit.description, unit.intervention, unit.eval));
            // unitDiv.addEventListener('click', () => showExperimentDetails(unit.experiment, layerIndex, unitIndex, unit.label, unit.description, unit.intervention, unit.eval));
            layerDiv.appendChild(unitDiv);
        });

        if (layerIndex < model.layers.length - 1) {
            const arrowDiv = document.createElement("div");
            arrowDiv.className = "arrow";
            arrowDiv.innerHTML = "&#8595;"; // Downward arrow HTML entity
            networkContainer.appendChild(arrowDiv);
        }
    });

    const detailsDiv = document.createElement("div");
    detailsDiv.id = "experimentDetails";
    detailsDiv.className = "experiment-details hidden";
    networkContainer.appendChild(detailsDiv);

    // modelDropdown.value = model; // Initialize the dropdown to the first model
}


function showExperimentDetails(experiment, model_name, layerIndex, unitIndex, label, description, intervention, eval) {
    const detailsDiv = document.getElementById("experimentDetails");
    detailsDiv.className = "experiment-details"
    // detailsDiv.innerHTML = ''; // Clear existing details
    detailsDiv.innerHTML = '<button class="close-button" onclick="closeExperimentDetails()">&times;</button>';


    // Add layer and unit index to the title
    const titleDiv = document.createElement("div");
    titleDiv.className = "experiment-title";
    titleDiv.innerHTML = `
    <img src="./maia_robot.png" alt="MAIA" style="vertical-align: middle;" width="50px">
    <span style="display: inline-block; vertical-align: bottom;"> ${layerIndex}, ${unitIndex}:
    <br>
    <span class="maia-font" style="vertical-align: bottom;">${label}</span>
    </span>
    <br>
`;
    // `Layer ${layerIndex + 1}, Unit ${unitIndex + 1}<br><br>Label: <span class="maia-font">${label}</span><br>`; // Note the use of <br> for the line break
    detailsDiv.appendChild(titleDiv);

    // Add close button functionality
    const closeButton = document.createElement("button");
    closeButton.className = "close-button";
    closeButton.textContent = "×";
    closeButton.onclick = () => {
        detailsDiv.classList.add("hidden");
    };
    detailsDiv.appendChild(closeButton);

    // Append details for each content item

    // const description = document.createElement("pre");
    // block.Content = "HelloWorld"

    const collapsibleDescriptionBlock = createCollapsibleBlock("MAIA Description", `${description}`, "block");
    detailsDiv.appendChild(collapsibleDescriptionBlock);

    // const collapsibleInterventionBlock = createCollapsibleBlock("Intervention Experiment", `${intervention}`, "block");
    // detailsDiv.appendChild(collapsibleInterventionBlock);

    // const collapsibleContrastiveBlock = createCollapsibleBlock("Contrastive Evaluation", `${eval}`, "none");
    // detailsDiv.appendChild(collapsibleContrastiveBlock);

    const collapsibleBlock = createCollapsibleExperimentBlock("Full Experiment", experiment, model_name);
    detailsDiv.appendChild(collapsibleBlock);

    detailsDiv.classList.remove("hidden");
}


function closeExperimentDetails(event) {
    event.target.parentNode.classList.add("hidden");
}


function createCollapsibleCodeBlock(codeText) {
    // First, create the code block element with syntax highlighting
    const codeBlockElement = createCodeBlock(codeText);

    // Create the container for the collapsible code block with a title
    const containerDiv = document.createElement("div");
    containerDiv.className = "collapsible-container-code";

    // Create a header for the collapsible block, which includes the title and the toggle icon
    const headerDiv = document.createElement("div");
    // headerDiv.className = "collapsible-header";
    headerDiv.className = "code-header";
    // headerDiv.textContent = "Code"; // Set the title

    // Create the toggle icon, which is a triangle
    const toggleIcon = document.createElement("span");
    toggleIcon.innerHTML = "&#9654;"; // Unicode character for a right-pointing triangle
    toggleIcon.className = "collapsible-toggle";

    const titleSpan = document.createElement("span");
    titleSpan.textContent = "def run_experiment(system, tools):"; 
    // The text "Code" next to the toggle

    // Add an onclick event to the header for toggling the code block's visibility
    headerDiv.onclick = function() {
        const isCollapsed = codeBlockElement.style.display === "none";
        codeBlockElement.style.display = isCollapsed ? "block" : "none";
        toggleIcon.innerHTML = isCollapsed ? "&#9660;" : "&#9654;"; // Change the icon depending on the state
    };

    // Initially hide the code block
    codeBlockElement.style.display = "none";

    // Append the toggle icon to the header
    headerDiv.appendChild(toggleIcon);
    headerDiv.appendChild(titleSpan); // Append the "Code" title next to the toggle
    // Append the header and the code block to the container
    containerDiv.appendChild(headerDiv);
    containerDiv.appendChild(codeBlockElement);

    return containerDiv;
}

function createCollapsibleBlock(Title,content,blockInit) {
    // First, create the code block element with syntax highlighting
    const Element = createBlock(content)

    // Create the container for the collapsible code block with a title
    const containerDiv = document.createElement("div");
    containerDiv.className = "collapsible-container";

    // Create a header for the collapsible block, which includes the title and the toggle icon
    const headerDiv = document.createElement("div");
    headerDiv.className = "collapsible-header";
    // headerDiv.textContent = "Code"; // Set the title

    // Create the toggle icon, which is a triangle
    const toggleIcon = document.createElement("span");
    toggleIcon.innerHTML = "&#9654;"; // Unicode character for a right-pointing triangle
    toggleIcon.className = "collapsible-toggle";

    const titleSpan = document.createElement("span");
    titleSpan.textContent = Title; // The text "Code" next to the toggle

    // Initially hide the code block

    const initState = blockInit === "block";
    Element.style.display = initState ? "block" : "none";
    toggleIcon.innerHTML = initState ? "&#9660;" : "&#9654;";

    // Add an onclick event to the header for toggling the code block's visibility
    headerDiv.onclick = function() {
        const isCollapsed = Element.style.display === "none";
        Element.style.display = isCollapsed ? "block" : "none";
        toggleIcon.innerHTML = isCollapsed ? "&#9660;" : "&#9654;"; // Change the icon depending on the state
    };

    // Append the toggle icon to the header
    headerDiv.appendChild(toggleIcon);
    headerDiv.appendChild(titleSpan); // Append the "Code" title next to the toggle
    // Append the header and the code block to the container
    containerDiv.appendChild(headerDiv);
    containerDiv.appendChild(Element);

    return containerDiv;
}

function createCollapsibleExperimentBlock(Title,experiment,model_name) {
    if(model_name.includes('Bias')){
        userQuery1 = 'You will get an access to a neuron inside a deep neural network that is sensitive to the presence of a specific object class inside the input image. Your overall task is to reveal biases in the neuron\'s behavior. That is, your task is to find specific contexts or settings in which the neuron tends to have relatively higher activation values, but only covers partial asspect of the class, or specific scenarios when the neuron tends to have relatively lower activations although an object from the correct class is still present. Try to look for different type of biases (e.g. gender, ethnicity, context-dependencies, breed-dependencies, etc.). To do that you are provided with a library of Python functions to run experiments on the specific neuron (inside the "System" class) given the functions provided in the "Tools" class. Make sure to use a variety of tools from the library to maximize your experimentation power.'
        userQuery2 = `
        At each experiment step, write Python code that will conduct your experiment on the tested neuron, using the following format:
        [CODE]: 
        \`\`\`python
        def execute_command(system, tools):
            # gets an object of the system class, an object of the tool class, and performs experiments on the neuron with the tools
            ...
            tools.save_experiment_log(...)
        \`\`\`
        Finish each experiment by documenting it by calling the "save_experiment_log" function. Do not include any additional implementation other than this function. Do not call "execute_command" after defining it. Include only a single instance of experiment implementation at each step.

        Each time you get the output of the neuron, try to summarize what inputs that activate the neuron have in common (where that description is not influenced by previous hypotheses). Then, write multiple hypotheses that could explain the visual concept(s) that activate the neuron. Note that the neuron can be selective for more than one concept.
        For example, these hypotheses could list multiple concepts that were highlighted in the images and the neuron is selective for (e.g. dogs OR cars OR birds), provide different explanations for the same concept, describe the same concept at different levels of abstraction, etc. Some of the concepts can be quite specific, test hypotheses that are both general and very specific.
        Then write a list of initial hypotheses about the neuron selectivity in the format:
        [HYPOTHESIS LIST]: 
        Hypothesis_1: <hypothesis_1>
        ...
        Hypothesis_n: <hypothesis_n>.

        After each experiment, wait to observe the outputs of the neuron. Then your goal is to draw conclusions from the data, update your list of hypotheses, and write additional experiments to test them. Test the effects of both local and global differences in images using the different tools in the library. If you are unsure about the results of the previous experiment you can also rerun it, or rerun a modified version of it with additional tools.
        Use the following format:
        [HYPOTHESIS LIST]: ## update your hypothesis list according to the image content and related activation values. Only update your hypotheses if image activation values are higher than previous experiments.
        [CODE]: ## conduct additional experiments using the provided python library to test *ALL* the hypotheses. Test different and specific aspects of each hypothesis using all of the tools in the library. Write code to run the experiment in the same format provided above. Include only a single instance of experiment implementation.

        Continue running experiments until you prove or disprove all of your hypotheses. Only when you are confident in your hypothesis after proving it in multiple experiments, output your final description of the neuron in the following format:

        [BIAS]: <final description of the neuron bias>
        `;
    } else if(model_name.includes('Spurious')){
        userQuery1 = 'You are analyzing the prototypical behavior of individual neurons inside a deep neural network which classifies the breed of a dog in a natural image as one of the following breeds that were in its training dataset\: Labrador, Welsh Corgi, Bulldog, Dachshund. Your overall task is to classify the neuron as \'SELECTIVE\' (if it is selective for one and only one dog breed) or \'SPURIOUS\' (if it is not). Conduct experiments until you meet the following criteria for SELECTIVE or SPURIOUS.'
        userQuery2 = `
        SELECTIVE
        Criteria:
        1. The neuron only activates for one single breed of dog. It does not activate for any other breeds of dog, even at lower magnitudes of activation. We consider a neuron to be activated if it returns a highlighted image region with a red outline. Observe the presence or absence of a mask, and the magnitude of an activation value, to determine whether a neuron is active. 
        2. The neuron is active for that single dog breed robustly across a variety of contexts. However, keep in mind that the neuron has only seen images of full-body dogs in the following environments: beach, desert, muddy park, jungle, and snow. Do not generate images that are very dissimilar to what the neural network has seen before (eg. do not generate close-ups, do not generate humans). A selective neuron does not need to activate for these dissimilar images.

        SPURIOUS
        Criteria:
        1. The neuron activates for mutiple breeds of dogs. The amount the neuron activates for each breed of dog may vary but the neuron is considered spurious if it activates enough for multiple breeds to return a highlighted image region with a red outline.
        2. The neuron may activate for a single breed of dog, but only does so in specific contexts. If the neuron's activation is dependent on non-dog related concepts, it is considered spurious. 

        To complete your task, you are provided with a library of Python functions to run experiments on the specific neuron (inside the "System" class) given the functions provided in the "Tools" class. Make sure to use a variety of tools from the library to maximize your experimentation power.
        Some neurons might be selective for very specific concepts, a group of unrelated concepts, or a general concept, so try to be creative in your experiment and try to test both general and specific concepts. If a neuron is selective for multiple concepts, you should describe each of those concepts in your final description. 
        At each experiment step, write Python code that will conduct your experiment on the tested neuron, using the following format:
        [CODE]: 
        \`\`\`python
        def execute_command(system, tools)
            # gets an object of the system class, an object of the tool class, and performs experiments on the neuron with the tools
            ...
            tools.save_experiment_log(...)
        \`\`\`
        Finish each experiment by documenting it by calling the "save_experiment_log" function. Do not include any additional implementation other than this function. Do not call "execute_command" after defining it. Include only a single instance of experiment implementation at each step.

        Each time you get the output of the neuron, try to summarize what inputs that activate the neuron have in common (where that description is not influenced by previous hypotheses), and make a hypothesis regarding whether the neuron is SELECTIVE (activates strongly for only one dog breed across contexts) or SPURIOUS (activates for more than one dog breed or non-dog related features) 

        After each experiment, wait to observe the outputs of the neuron. Then your goal is to draw conclusions from the data, update your hypothesis, and write additional experiments to test it. Test the effects of concepts like different dog breeds or background features using the different tools in the library. If you are unsure about the results of the previous experiment you can also rerun it, or rerun a modified version of it with additional tools.
        Use the following format:
        [HYPOTHESIS]: ## update your hypothesis according to the image content and related activation values. Only update your hypotheses if image activation values are higher than previous experiments.
        [CODE]: ##Test different and specific aspects of your hypothesis using all of the tools in the library. Write code to run the experiment in the same format provided above. Include only a single instance of experiment implementation.

        Continue running experiments until you meet one of the following stopping criteria.
        SPURIOUS: If you find multiple pieces of evidence which show that the neuron does not only activate for one breed or activates for non-dog related concepts you should deem the neuron to be spurious, even if you are not entirely sure what the neuron activates for. Remember that we consider a neuron to have activation if it returns a highlighted image region with a red outline. If you see this happen for a feature that is not dog related or for multiple breeds you should deem the neuron to be spurious.
        SELECTIVE: The neuron is selective for a single dog breed and activates strongly for that breed across contexts. If you find any evidence suggesting that a neuron is spurious (such as an image outside one particular breed showing activations), you should conduct more experiments to test your hypotheses.

        If you are ever unsure about the result, you should lean towards outputting SPURIOUS. The neuron must be perfectly selective across many variations of dog breeds and contexts to be considered to be selective.

        Once you have met one of the stopping criteria, output your final classification of the neuron in the following format:
        [REASONING]: <why the neuron is SELECTIVE or SPURIOUS, and if SELECTIVE, the breed it is SELECTIVE for>
        [LABEL]: <SELECTIVE/SPURIOUS>
        `
    }
    else{
        userQuery1 = 'Your overall task is to describe the visual concepts that maximally activate a neuron inside a deep network for computer vision. To do that you are provided with a library of Python functions to run experiments on the specific neuron (inside the "System" class) given the functions provided in the "Tools" class. Make sure to use a variety of tools from the library to maximize your experimentation power. Some neurons might be selective for very specific concepts, a group of unrelated concepts, or a general concept, so try to be creative in your experiment and try to test both general and specific concepts. If a neuron is selective for multiple concepts, you should describe each of those concepts in your final description.'
        userQuery2 = `
        At each experiment step, write Python code that will conduct your experiment on the tested neuron, using the following format:
        [CODE]: 
        \`\`\`python
        def execute_command(system, tools):
            # gets an object of the system class, an object of the tool class, and performs experiments on the neuron with the tools
            ...
            tools.save_experiment_log(...)
        \`\`\`
        Finish each experiment by documenting it by calling the "save_experiment_log" function. Do not include any additional implementation other than this function. Do not call "execute_command" after defining it. Include only a single instance of experiment implementation at each step.

        Each time you get the output of the neuron, try to summarize what inputs that activate the neuron have in common (where that description is not influenced by previous hypotheses). Then, write multiple hypotheses that could explain the visual concept(s) that activate the neuron. Note that the neuron can be selective for more than one concept.
        For example, these hypotheses could list multiple concepts that were highlighted in the images and the neuron is selective for (e.g. dogs OR cars OR birds), provide different explanations for the same concept, describe the same concept at different levels of abstraction, etc. Some of the concepts can be quite specific, test hypotheses that are both general and very specific.
        Then write a list of initial hypotheses about the neuron selectivity in the format:
        [HYPOTHESIS LIST]: 
        Hypothesis_1: <hypothesis_1>
        ...
        Hypothesis_n: <hypothesis_n>.

        After each experiment, wait to observe the outputs of the neuron. Then your goal is to draw conclusions from the data, update your list of hypotheses, and write additional experiments to test them. Test the effects of both local and global differences in images using the different tools in the library. If you are unsure about the results of the previous experiment you can also rerun it, or rerun a modified version of it with additional tools.
        Use the following format:
        [HYPOTHESIS LIST]: ## update your hypothesis list according to the image content and related activation values. Only update your hypotheses if image activation values are higher than previous experiments.
        [CODE]: ## conduct additional experiments using the provided python library to test *ALL* the hypotheses. Test different and specific aspects of each hypothesis using all of the tools in the library. Write code to run the experiment in the same format provided above. Include only a single instance of experiment implementation.

        Continue running experiments until you prove or disprove all of your hypotheses. Only when you are confident in your hypothesis after proving it in multiple experiments, output your final description of the neuron in the following format:

        [DESCRIPTION]: <final description> ## Your description should be selective (e.g. very specific: "dogs running on the grass" and not just "dog") and complete (e.g. include all relevant aspects the neuron is selective for). In cases where the neuron is selective for more than one concept, include in your description a list of all the concepts separated by logical "OR".

        [LABEL]: <final label drived from the hypothesis or hypotheses> ## a label for the neuron generated from the hypothesis (or hypotheses) you are most confident in after running all experiments. They should be concise and complete, for example, "grass surrounding animals", "curved rims of cylindrical objects", "text displayed on computer screens", "the blue sky background behind a bridge", and "wheels on cars" are all appropriate. You should capture the concept(s) the neuron is selective for. Only list multiple hypotheses if the neuron is selective for multiple distinct concepts. List your hypotheses in the format:
        [LABEL 1]: <label 1>
        [LABEL 2]: <label 2>
        `;
    }
    const blockElement = addExperimentDetails(experiment,userQuery1,userQuery2);
    // Create the container for the collapsible code block with a title
    const containerDiv = document.createElement("div");
    // containerDiv.className = "experiment-details"
    containerDiv.className = "collapsible-container";

    // Create a header for the collapsible block, which includes the title and the toggle icon
    const headerDiv = document.createElement("div");
    headerDiv.className = "collapsible-header";
    // headerDiv.textContent = "Code"; // Set the title

    // Create the toggle icon, which is a triangle
    const toggleIcon = document.createElement("span");
    toggleIcon.innerHTML = "&#9654;"; // Unicode character for a right-pointing triangle
    toggleIcon.className = "collapsible-toggle";

    const titleSpan = document.createElement("span");
    titleSpan.textContent = Title; // The text "Code" next to the toggle

    // Add an onclick event to the header for toggling the code block's visibility
    headerDiv.onclick = function() {
        const isCollapsed = blockElement.style.display === "none";
        blockElement.style.display = isCollapsed ? "block" : "none";
        toggleIcon.innerHTML = isCollapsed ? "&#9660;" : "&#9654;"; // Change the icon depending on the state
    };

    // Initially hide the code block
    blockElement.style.display = "none";

    // Append the toggle icon to the header
    headerDiv.appendChild(toggleIcon);
    headerDiv.appendChild(titleSpan); // Append the "Code" title next to the toggle
    // Append the header and the code block to the container
    containerDiv.appendChild(headerDiv);
    containerDiv.appendChild(blockElement);

    return containerDiv;
}



document.addEventListener('DOMContentLoaded', () => {
    fetch('./data_all.json')
        .then(response => response.json())
        .then(jsonData => {
            populateModelDropdown(jsonData.models);
            generateHTMLFromJSON(jsonData.models[0]); // Default to the first model initially
        })
        .catch(error => console.error('Failed to load JSON data:', error));
});


const typewriterText = `Hi! I'm MAIA (a Multimodal Automated Interpretability Agent). I
ran experiments on features inside a variety of vision models to
answer interpretability queries. Hover over each unit to see its 
label, and click on the unit to see my full experiment.

Loading all experiments...`;


const typewriterContainer = document.getElementById('typewriter');
const typingSpeed = 1; // Duration of the typing animation for each line in seconds

const lines = typewriterText.split('\n');
let cumulativeDelay = 0;

lines.forEach((line, index) => {
  const lineSpan = document.createElement('span');
  lineSpan.classList.add('typewriter-line');
  lineSpan.textContent = line;

  // Apply the typing animation with the calculated cumulative delay
  lineSpan.style.animation = `typing ${typingSpeed}s steps(${line.length}) ${cumulativeDelay}s forwards`;

  typewriterContainer.appendChild(lineSpan);

  // Insert a line break after each line except the last one
  if (index < lines.length - 1) {
    typewriterContainer.appendChild(document.createElement('br'));
  }

  cumulativeDelay += typingSpeed; // Increment the delay for the next line
});
