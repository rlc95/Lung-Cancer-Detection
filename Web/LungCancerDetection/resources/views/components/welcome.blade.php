<div class="p-6 lg:p-8 bg-white border-b border-gray-200">
    <div class="row">
        <div class="col">
            <x-application-logo class="block h-12 w-auto" />
        </div>
        <div class="col">
            <h2 style="position: relative; right: 530px; margin-top: 18px; font-weight: bold; font-size: 20px;">Lung_Cancer_Detection_System</h2>
        </div>
    </div>
    
    <h1 class="mt-8 text-2xl font-medium text-gray-900">
        Welcome to Lung Cancer Detection!
    </h1>

    <p class="mt-6 text-gray-500 leading-relaxed">
        Some lung cancers can be found by screening, but most lung cancers are found because they are causing problems. The actual diagnosis of lung cancer is made by looking at a sample of lung cells in the lab. If you have possible signs or symptoms of lung cancer, see your doctor.
    </p>
</div>

<div class="bg-gray-200 bg-opacity-25 grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8 p-6 lg:p-8">
    <div>
    
        <div class="card">
            <div class="card-header">
            <h3 class="card-title">Lung Cancer Detection Tool</h3>
            </div>
            <div class="card-body">
            <div>
                <form action="/predict" name="mfrm" id="mfrm" method="post" autocomplete="off" enctype="multipart/form-data">
                    <div class="form-row">
                        <!--     
                        <input type="file" id="myFile" name="filename">
                        -->
                        <input type="file" accept="image/*" id="myFile" name="filename" onchange="loadFile(event)">
                        <p><img id="output" width="200"/></p>
    
                    </div>
            
                    <div class="form-row">
                                
                        <div class="form-group col-md-8">
                            <button type="submit" class="btn btn-primary">Submit</button>
                        </div>
                
                    </div>
                </form>
            </div>
            </div>
            
        </div>
    
    </div>

    <div>
        <p></p>
    </div>

    <div>
        
    </div>

    <div>
        
    </div>
</div>


<script>
    var loadFile = function(event) {
        var image = document.getElementById('output');
        image.src=URL.createObjectURL(event.target.files[0]);
    };


    $(function() {

    $('#submitbtn').click(function(e) {
    if ($('#mfrm').valid()) {
        
        document.forms["mfrm"].submit();
        alert("Register Process was done successfully");
    }
    else {
        
        return false;
    }
    });

    });

</script>