

/*! ******************************

  Handlebars helpers

  *******************************/



/*

 * usage

 *

{{#hcList gridList begin="0" end="20"}}

  <div>

    ... {{name}} ...

    ... {{cardNum}} ...

  </div>

{{/hcList}}

 */



Handlebars.registerHelper("hcList", function(context, option) {

    var out = "", data, optItem;

    var convertOptions = (option.hash.template) ? window[option.hash.template] : {};

    

    var addOpt = {};

    addOpt = $.extend(true, addOpt, convertOptions);

  

    var hasConvertOpt = false;

    

    var begin = parseInt(option.hash.begin) || 0;

    var end = parseInt(option.hash.end) || context.length;

    var i = (begin < context.length) ? begin : (begin > context.length)? context.length : 0;

    var j = (end  > begin && end <= context.length) ?  end : context.length;



    for ( ; i < j; i++) {

        var temp = {};

        if (option.data) {

            option.data.index = i;     //@index

            option.data.left = (end-1)-i;

            option.data.first = (i === begin);

            option.data.last  = (i === (end-1));

            

            data = Handlebars.createFrame(option.data);

        }

/*

        $.each(context[i], function(key){

            optItem = convertOptions[key];

            if(optItem) {

                if(_.isArray(optItem)) {

                    $.each(optItem, function(index, item) {

                        $.hc.convertFunc(key, item, context[i]);

                    });

                } else if(_.isObject(optItem)) {

                    $.hc.convertFunc(key, optItem, context[i]);

                }

            }

        });

*/        

            temp = $.extend(true, temp, context[i]); // temp에 context 값 복사

           

            $.each(context[i], function(contKey){  // 원본데이터의 키값을 가져옴

                optItem = convertOptions[contKey];  // 원본데이터의 키 값으로 convert value 호출

                if( typeof optItem != "undefined") {    // convert value가 있으면

                    hasConvertOpt = true;   

                    if(_.isArray(optItem)) {            // array 이면 (여러개)

                        $.each(optItem, function(index, item) { // 여러번 호출

                            $.hc.convertFunc(contKey, item, context[i]);

                        });

                    } else if(_.isObject(optItem)) {    // object이면 (한개)

                        $.hc.convertFunc(contKey, optItem, context[i]); // 한번 호출

                    }

                    addOpt = _.omit(addOpt, contKey);  // convert key에서 context 키를 삭제 

                }

            });

            

            // context에 지정된 키가 없는 경우 convertOpt내부의 키로 맵핑한다.

            var tmpi = 0;

            if(hasConvertOpt) {

                $.each(addOpt, function(optKey){

                    $.hc.convertFunc(optKey, addOpt[optKey], context[i], temp);

                });

            }

   

        out += option.fn($.extend({}, context[i],{leftLength: context.length-i}), { data : data });



        flag = true;

    }

/*

    return context.map(function(item,index) {

        return option.fn(item);

    }).join('');

*/

    return out;

});





/**

 * 금액 포맷팅

 * usage: {{getMoneyFormat money_value}}

 */

Handlebars.registerHelper('getMoneyFormat', function(context) {

    return $.hc.getMoneyFormat(context);

});



/**

 * If Equals

 * if_eq this compare=that

 */

Handlebars.registerHelper('if_eq', function(context, options) {

    if (context == options.hash.compare)

        return options.fn(this);

    return options.inverse(this);

});





Handlebars.registerHelper('great_than', function(context, options) {

    if (context < options.hash.compare)

        return options.fn(this);

    return options.inverse(this);

});



Handlebars.registerHelper('less_than', function(context, options) {

    if (context > options.hash.compare)

        return options.fn(this);

    return options.inverse(this);

});

Handlebars.registerHelper('more_than', function(context, options) {

    if (context >= parseInt(options.hash.compare))

        return options.fn(this);

    return options.inverse(this);

});





/**

 * Unless Equals

 * unless_eq this compare=that

 */

Handlebars.registerHelper('unless_eq', function(context, options) {

    if (context == options.hash.compare)

        return options.inverse(this);

    return options.fn(this);

});





Handlebars.registerHelper('safeString', function(context) {

    return new Handlebars.SafeString(context);

});



/*



// debug helper

// usage: {{debug}} or {{debug someValue}}

// from: @commondream (http://thinkvitamin.com/code/handlebars-js-part-3-tips-and-tricks/)

Handlebars.registerHelper("debug", function(optionalValue) {

  console.log("Current Context");

  console.log("====================");

  console.log(this);



  if (optionalValue) {

    console.log("Value");

    console.log("====================");

    console.log(optionalValue);

  }

});





//  return the first item of a list only

// usage: {{#first items}}{{name}}{{/first}}

Handlebars.registerHelper('first', function(context, block) {

  return block(context[0]);

});







// a iterate over a specific portion of a list.

// usage: {{#slice items offset="1" limit="5"}}{{name}}{{/slice}} : items 1 thru 6

// usage: {{#slice items limit="10"}}{{name}}{{/slice}} : items 0 thru 9

// usage: {{#slice items offset="3"}}{{name}}{{/slice}} : items 3 thru context.length

// defaults are offset=0, limit=5

// todo: combine parameters into single string like python or ruby slice ("start:length" or "start,length")

Handlebars.registerHelper('slice', function(context, block) {

  var ret = "",

      offset = parseInt(block.hash.offset) || 0,

      limit = parseInt(block.hash.limit) || 5,

      i = (offset < context.length) ? offset : 0,

      j = ((limit + offset) < context.length) ? (limit + offset) : context.length;



  for(i,j; i<j; i++) {

    ret += block(context[i]);

  }



  return ret;

});









//  return a comma-serperated list from an iterable object

// usage: {{#toSentance tags}}{{name}}{{/toSentance}}

Handlebars.registerHelper('toSentance', function(context, block) {

  var ret = "";

  for(var i=0, j=context.length; i<j; i++) {

    ret = ret + block(context[i]);

    if (i<j-1) {

      ret = ret + ", ";

    };

  }

  return ret;

});







//  format an ISO date using Moment.js

//  http://momentjs.com/

//  moment syntax example: moment(Date("2011-07-18T15:50:52")).format("MMMM YYYY")

//  usage: {{dateFormat creation_date format="MMMM YYYY"}}

Handlebars.registerHelper('dateFormat', function(context, block) {

    if (window.moment && context && moment(context).isValid()) {

            var f = block.hash.format || "MMM Do, YYYY";

            return moment(context).format(f);

    }else{

            return context;   //  moment plugin is not available, context does not have a truthy value, or context is not a valid date

    }

});



*/



//--------------------------------------------------------------------



/**

 * usages (handlebars)

 * {{addition 10 to=10}}

 * {{addition this to=20}}

 * {{addition this to="-5"}}

**/

/*

Handlebars.registerHelper('addition', function(context, options){

    return context + parseFloat(options.hash.to);

});

*/



/**

 * usages (handlebars)

 * {{short_string this}}

 * {{short_string this length=150}}

 * {{short_string this length=150 trailing="---"}}

**/

/*

Handlebars.registerHelper('short_string', function(context, options){

    //console.log(options);

    var maxLength = options.hash.length || 100;

    var trailingString = options.hash.trailing || '';

    if(context.length > maxLength){

        return context.substring(0, maxLength) + trailingString;

    }

    return context;

});

*/



//--------------------------------------------------------------------





// HELPER: #key_value

//

// Usage: {{#key_value obj}} Key: {{key}} // Value: {{value}} {{/key_value}}

//

// Iterate over an object, setting 'key' and 'value' for each property in

// the object.

/*

Handlebars.registerHelper("key_value", function(obj, fn) {

    var buffer = "",

        key;



    for (key in obj) {

        if (obj.hasOwnProperty(key)) {

            buffer += fn({key: key, value: obj[key]});

        }

    }



    return buffer;

});

*/

// HELPER: #each_with_key

//

// Usage: {{#each_with_key container key="myKey"}}...{{/each_with_key}}

//

// Iterate over an object containing other objects. Each

// inner object will be used in turn, with an added key ("myKey")

// set to the value of the inner object's key in the container.

/*

Handlebars.registerHelper("each_with_key", function(obj, fn) {

    var context,

        buffer = "",

        key,

        keyName = fn.hash.key;



    for (key in obj) {

        if (obj.hasOwnProperty(key)) {

            context = obj[key];



            if (keyName) {

                context[keyName] = key;

            }



            buffer += fn(context);

        }

    }



    return buffer;

});

*/

//------------------------------------------------------------------



/*

 * usage

 *

{{#stripes myArray "even" "odd"}}

  <div class="{{stripeClass}}">

    ... code for the row ...

  </div>

{{else}}

  <em>There aren't any people.</em>

{{/stripes}}

 */

/*

Handlebars.registerHelper("stripes", function(array, even, odd, fn, elseFn) {

    if (array && array.length > 0) {

      var buffer = "";

      for (var i = 0, j = array.length; i < j; i++) {

        var item = array[i];



        // we'll just put the appropriate stripe class name onto the item for now

        item.stripeClass = (i % 2 == 0 ? even : odd);



        // show the inside of the block

        buffer += fn(item);

      }



      // return the finished buffer

      return buffer;

    }

    else {

      return elseFn();

    }

  });

*/





//-------------------------------------------------------------------



/*

 * usage



            {{#list gridList a=111 b="bbbbbb" }}

            <tr>

                <td>{{@index}}</td>

                <td>{{id}}</td>

                <td>{{name}}</td>

                <td>{{age}}</td>

                <td>{{address}}</td>

            </tr>

            {{/list}}



 */

/*

Handlebars.registerHelper('list', function(context, options) {



    var out = "", data;



    for ( var i = 0; i < context.length; i++) {



        if (options.data) {

            options.data.index = i;     //@index

            data = Handlebars.createFrame(options.data);

        }



        out += options.fn(context[i], {

            data : data

        });

    }



    return out;

});

*/

//--------------------------------------------------------------------
