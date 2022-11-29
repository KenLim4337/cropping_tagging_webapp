$(document).ready(() => {
    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }

    // required by Django
    var csrftoken = $("[name=csrfmiddlewaretoken]").val();
    
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
    
    $("#image-input").on('change', function(){
        const reader = new FileReader();
        reader.onload = () => {
            const uploaded_image = reader.result;
            $("#display-image").attr('src', uploaded_image);
        };
        reader.readAsDataURL(this.files[0]);
    });


    var jcp;

    Jcrop.load('display-image').then(img => {
        // You can enable multiple cropping with this line:
        jcp = Jcrop.attach(img, { multi: true });
    });


    $('#save').on('click', () => {
        // we check if at least one crop is available
        if (jcp.active) {
            // we are looping cropped areas
            
            $('#crop-data').empty()

            img_data = $('#display-image').attr('src').split('base64,')[1]

            areas = []

            for (area of jcp.crops) {
                x1 = area.pos['x']
                x2 = area.pos['x'] + area.pos['w']
                y1 = area.pos['y']
                y2 = area.pos['y'] + area.pos['h']

                $('#crop-data').append('<div>BBox: x1 = ' + x1 + ', x2 = ' + x2 + ', y1 = ' + y1 + ', y2 = ' + y2 + '</div>')

                areas.push([x1,x2,y1,y2])

               
            }

            // API call to run searches

            // API call returns list of image links, which we then pass to bulk_image_generator

            $.ajax({
                url: 'http://localhost:8000/api/search/', 
                type: 'POST',
                data: {
                    'api_key': 123456,
                    'areas': areas,
                    'img': img_data
                },
            }).done(function( data ) {
                if(data.success == 'success') {
                    console.log(data.results)
                    $('#image-results').html(bulk_image_generator(data.results));
                }
            });
        }
    })
});

function bulk_image_generator(src_list) {
    result = $('<div class="result"></div>')

    for (var i=0; i < Object.keys(src_list).length; i++) {
        photoId = src_list[i]['photo_id'];
        photoUrl = src_list[i]['url'];
        distance = src_list[i]['hog_distance'];

        result.append('<div class="result-photo"><div class="photo-header">Photo '+photoId+'</div><img src="'+ photoUrl[0] +'" /><div class="info"><div class="hog">Hog: '+distance+'</div></div></div>')
    }

    return result
}