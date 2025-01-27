<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class LungCancerDetection extends Controller
{
    public function imageReqst(Request $request){

        

    }

    public function predictResult(Request $request){ 

        return view('predict');
        /*
        ->with([
            'survey' => $survey
             
        ]);
        */
    }
}


