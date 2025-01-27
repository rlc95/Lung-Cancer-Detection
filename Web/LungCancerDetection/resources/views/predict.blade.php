<x-app-layout>
    <x-slot name="header">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            {{ __('Dashboard') }}
        </h2>
    </x-slot>

    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="bg-white overflow-hidden shadow-xl sm:rounded-lg">
                <div class="container">
                    <div class="card">
                        <div class="card-body">

                            <div class="table-responsive">
                                
                                    <table class="table table-striped table-sm w-auto"  id="Tble" > 
                            
                                        <tr> 

                                            <th class='table-coloum-width'>
                                                <img src="{{asset('Normal.jpg')}}" class="d-block image fluid" style="width: 180px; height: 180px;" alt="Logo">
                                            </th>
                                            <td colspan="3">
                                                <img src="{{asset('norm.png')}}" class="d-block image fluid" style="width: 35px; height: 35px;" alt="Logo">
                                            </td>
                                            
                                        </tr>
                                        <tr> 
                                            <th class='table-coloum-width'>Cancer</th>
                                            <td colspan="3" style="background-color: rgb(255, 255, 255);">Not Detected</td>
                                        </tr>
                            
                                        <tr>
                                            <th>Cancer Position</th>
                                            <td colspan="3">Normal</td>
                                        </tr>

                                        <tr>
                                            <th>Instructions</th>
                                            <td colspan="3">You don’t have lung cancer..!<br>
                                                            Don’t smoke or quit smoking if you do<br>
                                                            Avoid second hand smoke and other substances that can harm your lungs<br>
                                                            Eat a healthy diet and maintain a weight that’s healthy for you<br>
                                                            Suggest that eating fruits and vegetables (two to six-and-a-half cups per day)
                                            </td>
                                        </tr>
                        
                                    </table> 
                                
                            </div>


                        </div>

                    </div>
                </div>
            </div>
            
        </div>
    </div>
</x-app-layout>