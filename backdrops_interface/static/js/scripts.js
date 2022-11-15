
$(document).ready(() => {
    const image_input = document.querySelector("#image-input");

    image_input.addEventListener("change", function() {
        const reader = new FileReader();
        reader.addEventListener("load", () => {
            const uploaded_image = reader.result;
            document.querySelector("#display-image").src = uploaded_image;
        });
        reader.readAsDataURL(this.files[0]);
    });

    var jcp;

    Jcrop.load('display-image').then(img => {
        //You can enable multiple cropping with this line:
        jcp = Jcrop.attach(img, { multi: true });
    });

    var link = document.getElementById('save');
    link.onclick = function () {
        //we check if at least one crop is available
        if (jcp.active) {
            var i = 0;
            var fullImg = document.getElementById("target");
            //we are looping cropped areas
            for (area of jcp.crops) {
                i++;
                
                console.log(area.pos)
            }
        }
    };
});