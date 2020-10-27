

var max_images = 20; // max pre-loaded on screen


async function remove_first_image(){
    images = document.getElementById("images").children;
    if(images.length > 0){
        images[0].remove();        
    }    
    big_image = document.getElementById("big_image");
    big_image.setAttribute("src", "");
    
    focus_first_image();
}

async function focus_first_image(){
    images = document.getElementById("images").children;
    if(images.length > 0){
        images[0].setAttribute("class", "image focus_image");
    }
}


async function add_annotation(is_bicycle){
    images = document.getElementById("images").children;
    if(images.length > 0){
        url = images[0].alt; 
        label = images[0].getAttribute("label");
        
        var annotation_ok = true
        if(label != ""){
            if(label.toString() == "0" && is_bicycle){
                alert("Warning: this image does *not* contain a bicycle, according to existing annotations");
                annotation_ok = false;
            }
            else if(label.toString() == "1" && !is_bicycle){
                 alert("Warning: this image *does* contain a bicycle, according to existing annotations");          
                 annotation_ok = false
            }
        }
        if(annotation_ok){
            eel.add_annotation(url, is_bicycle)(); // Call to python function
            remove_first_image();
        }        
    } 
    
    // 
}

async function training_loaded(){
    return await eel.training_loaded()();
}

async function validation_loaded(){
    return await eel.validation_loaded()();
}



setInterval(async function(){ 
    //check for updated accuracy scores every 10 seconds
    processing_time = await eel.estimate_processing_time()();
    time_div = document.getElementById("time");
    
    console.log("processing time: "+processing_time.toString());

    
    if(processing_time > 0){
        if(processing_time < 90){
            message = Math.floor(processing_time).toString()+" seconds ";
        }
        else if(processing_time < 240){
            message = (Math.round(processing_time/30)/2).toString()+" minutes ";
        }
        else if(processing_time < 600){
            message = Math.round(processing_time/60).toString()+" minutes ... maybe get a cup of coffee ";
        }
        else if(processing_time < 5400){
            message = Math.round(processing_time/60).toString()+" minutes ... maybe take a short break and get some exercise ";            
        }
        else{
            message = (Math.round(processing_time/1800)/2).toString()+" hours ... maybe have a meal and come back later ";            
        }
        
        time_div.style="visibility:visible";
        time_div.innerHTML = '<b>Estimated Time remaining to prepare annotated images for machine learning (download and extract COCO and ImageNet vectors):</b><br /> '+message; 
    }
    else{
        time_div.style="visibility:hidden";
    }
    
}, 10000);




setInterval(async function(){ 
    //check for updated accuracy scores every 5 seconds
    accuracies = await eel.get_current_accuracies()();
    console.log("accuracies: "+accuracies.toString());
    if(accuracies.length > 0){
        stats = document.getElementById("stats")
        stats.style="visibility:visible";
        fscore = accuracies[0];
        if(fscore > 0){
            stats.innerHTML = 'Target Accuracy: F-Score = 0.85 <br />Current Accuracy: F-Score = '+fscore.toString();
        }
    }
    
}, 5000);


setInterval(async function(){ 
    //check for new images to annotate every half second    

    if(!validation_loaded()){        
        return false;
    }
    else{
        document.getElementById("instructions").style="visibility:visible";
    }
    if(!training_loaded){
        // TODO: MESSAGE ABOUT PRACTICE MODE?
    }

    images = document.getElementById("images");

    current = images.children.length;

    for(var i = current; i <= max_images; i++){
        image_details = await eel.get_next_image()(); // Call to python function
        if(image_details == null || image_details.length == 0){
            break;
        }
        image_url = image_details[0];
        image_thumbnail = image_details[1];
        image_label = image_details[2];
        
        new_image = document.createElement("IMG");
        new_image.setAttribute("src", image_thumbnail);
        new_image.setAttribute("alt", image_url);
        new_image.setAttribute("class", "image");
        new_image.setAttribute("label", image_label.toString());
         
        if(document.getElementById("images").children.length <= max_images){
            document.getElementById("images").appendChild(new_image);
        }
        else{
            new_image.remove(); // race condition: we are already ok
            break;
        }
    }    
    focus_first_image(); 
}, 500);



// LOG ANNOTATIONS
document.addEventListener("keypress", function onEvent(event) {
    //console.log(event);
    if (event.key === "b") {
        add_annotation(true);
    }
    else if (event.key === "n") {
        add_annotation(false);
    }
    else if (event.key === "z") {
        images = document.getElementById("images").children;
        if(images.length > 0){
            big_image = document.getElementById("big_image");
            url = images[0].alt;
            big_image.setAttribute("src", url);
        }
    }
    
});






        