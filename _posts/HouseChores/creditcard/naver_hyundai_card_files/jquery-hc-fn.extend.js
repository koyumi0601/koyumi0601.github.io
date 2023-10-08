﻿﻿(function($){
    $.fn.extend({

        /**
         * 날짜 입력 필드에 특정 기간의 날짜를 보여준다.
         * selector : radio button name 
         * initTerms : 초기에 보여줘야 할 날짜 기간 (week, one_month, two_month...) 
         * startInput : 시작일이 들어갈 input id 
         * endInput : 종료일이 들어갈 input id 
         * autoSet: 로딩 시 초기값 셋팅 
         * 예) 
         * $("use_day").hcDateTerm({startInput:"searchDateStart", // 시작일 id 
         *                          endInput:"searchDateEnd" // 종료일 id });
         */
        hcDateTerm : 
        function(o) {
            var options = {
              initTerms:"week",
              autoSet:true,
              startInput:"",
              endInput:""
            };
                      
            var  dateTerms = {
                 today:0,
                 week: -7,
                 one_month: "-1",
                 two_month: "-2",
                 three_month: "-3",
                 six_month: "-6",
                 twelve_month:"-12"
            };
             
            var dateObj = new Date();
            var config = $.extend({}, options, o);
                       
            if(config.autoSet) {
                var _this="";
                $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[config.initTerms]));
                
                $(":input[name="+ $(this).selector+"]").filter(function(){
                        if(this.value == config.initTerms){
                            _this = $(this);
                            return true;
                        }
                    }).prop("checked", true).siblings("label[for="+_this.attr("id")+"]").addClass("on");
            }
              
            $(":input[name="+ $(this).selector+"]").on("click", function(ev){
                var dateObj = new Date();
                $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[$(this).val()]));
               
                $("#"+config.startInput).removeClass("error");
                $("#"+config.endInput).removeClass("error");
                
                $("#"+config.endInput).siblings("p").html("");
                
            });
        },
        /**
         * 날짜 입력 필드에 특정 기간의 날짜를 보여준다.
         * selector : select box name 
         * initTerms : 초기에 보여줘야 할 날짜 기간 (week, one_month, two_month...) 
         * startInput : 시작일이 들어갈 input id 
         * endInput : 종료일이 들어갈 input id 
         * autoSet: 로딩 시 초기값 셋팅 
         * 예) 
         * $("use_day").hcSelectDateTerm({startInput:"searchDateStart", // 시작일 id 
         *                          endInput:"searchDateEnd" // 종료일 id });
         */
        hcSelectDateTerm : 
        function(o) {
            var options = {
              initTerms:"week",
              autoSet:true,
              startInput:"",
              endInput:""
            };
                      
            var  dateTerms = {
                 today:0,
                 week: -7,
                 one_month: "-1",
                 two_month: "-2",
                 three_month: "-3",
                 six_month: "-6"
            };
             
            var dateObj = new Date();
            var config = $.extend({}, options, o);
                       
            if(config.autoSet) {
                var _this="";
                $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[config.initTerms]));
            }
            
            $(":input[name="+ $(this).selector+"]").on("change", function(ev){
                if($(this).val() == "today" 
                    || $(this).val() == "week" 
                    || $(this).val() == "one_month" 
                    || $(this).val() == "two_month"
                    || $(this).val() == "three_month" 
                    || $(this).val() == "six_month") {
                    
                    var dateObj = new Date();
                    $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                    $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[$(this).val()]));
                   
                    $("#"+config.startInput).removeClass("error");
                    $("#"+config.endInput).removeClass("error");
                    
                    $("#"+config.endInput).siblings("p").html("");
                    
                }
            });
        },
        
        /**
         * 날짜 입력 필드에 특정 기간의 날짜를 보여준다. (직접입력 선택 버튼 추가)
         * autoSet : 초기에 날짜를 보여줄지 여부 (true, false) 
         * initTerms : 초기에 보여줘야 할 날짜 기간 (week, one_month, two_month...) 
         * startInput : 시작일이 들어갈 input id   
         * endInput : 종료일이 들어갈 input id     
         * autoSet: 로딩 시 초기값 셋팅 
         * dateBoxId : 직접입력 날짜박스가 있는 div (화면노출여부 처리용)
         * userMonth : 직접입력 선택시 보여줘야 할 날짜 기간 (week, one_month, two_month...) 
         * 예) 
         *     $("searchTerm").hcDateTermPlus({
         *                        autoSet : "${autoSet}",
         *                        initTerms : "${initTerms}",
         *                        startInput : "otaDt1",
         *                        endInput : "otaDt2",
         *                        dateBoxId : "inspayperiod_inputdate",
         *                        userMonth : "week"
         *                        });
         */
        hcDateTermPlus : 
        function(o) {
            var options = {
              initTerms:"week",
              autoSet:true,
              startInput:"",
              endInput:"",
              dateBoxId:"",
              userMonth:"today"
            };
                      
            var  dateTerms = { 
                 today:0,
                 week: -7,
                 one_month: "-1",
                 two_month: "-2",
                 three_month: "-3",
                 six_month: "-6",
                 twelve_month:"-12"
            };
             
            var dateObj = new Date();
            var config = $.extend({}, options, o);
                       
            if(config.autoSet) {
                var _this="";
                if ( config.initTerms == 'user_month' ) { //직접입력
                    if ( config.dateBoxId != '' ) {
                        if ( $("#"+config.endInput).val() == "" ) {
                            $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                            $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[config.userMonth]));
                        }
                    }
                } else {
                    $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                    $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[config.initTerms]));
                }
                $(":input[name="+ $(this).selector+"]").filter(function(){
                        if(this.value == config.initTerms){
                            _this = $(this);
                            return true;
                        }
                    }).prop("checked", true);
            }
            
            if ( config.dateBoxId != '' ) {
                if ( config.initTerms == 'user_month' ) { //직접입력
                    $("#"+config.dateBoxId).show();
                } else {
                    $("#"+config.dateBoxId).hide();
                }      
            }   
            
            $(":input[name="+ $(this).selector+"]").on("click", function(ev){
                var dateObj = new Date(); 
                $(this).prop("checked", true);
                if ($(this).val() == 'user_month') { //직접입력
                    $("#"+config.dateBoxId).show();
                    $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                    $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[config.userMonth]));   
                } else {
                    $("#"+config.dateBoxId).hide();
                    $("#"+config.endInput).val($.hc.date.currDate(dateObj));
                    $("#"+config.startInput).val($.hc.date.addDate(dateObj, dateTerms[$(this).val()]));
                }
                $("#"+config.startInput).removeClass("error");
                $("#"+config.endInput).removeClass("error");
                $("#"+config.endInput).siblings("p").html("");
            });
        },            
        /**
         * 이용안내 show/hide
         * 클릭 이벤트에 show_hide 클래스 추가 
         * 토글 대상 div에 toggleDiv 클래스 추가하여 사용해 주세요 
         * 예)
         * $('.show_hide').hcShowHide();
         */
        hcShowHide : 
        function (options) {
            var defaults = {
                    speed: 500
            };
            var options = $.extend(defaults, options);

            $(this).on("click", function (e) {
                var toggleDiv = $(this).attr('href');
                $(toggleDiv).slideToggle(options.speed);
                e.preventDefault();
            });
        },

        /**
         * check box 컨트롤
         * selector : 다른 checkbox를 컨트롤(check on/off) 할 부모 checkbox id 
         * 부모의 child attribute를 자식의 name과 동일하게 하는 설정이 필요함. 
         * <input type="checkbox" name="parent" id="chk_all" child="체크해야 할 자식 name (여러개일 경우 콤마(,)로 구분" />
         * 예) 
         * $('#chk_all').hcCheckAll();
         */
        hcCheckAll:
        function(callback) {
            var _this = $(this);
            var flag = false;
            var name = [];
            var selector = "";
            if(_this.attr("child").indexOf(",") != -1) {
                name = _this.attr("child").split(",");
            }else {
                name[0] = _this.attr("child");
            }
            
            for(i = 0 ; i < name.length; i++) {
                selector += (i == 0)? "input:checkbox" : ", input:checkbox";
                selector += "[name="+name[i]+"]";
            }
            
            // 자식 checkbox 체크 처리
            _checked = function() {
                var checked = _this.prop("checked") || false;
                //for(i = 0 ; i < name.length; i++) {
                //    $.each($(":input[name="+name[i]+"]"), function(i) {
                      $(selector).each(function(i) {  
                        if(!$(this).is(':disabled')) {
                            if($(this).get(0) != _this.get(0)) {
                                if(!checked) {
                                    $(this).prop("checked", checked).siblings('label').removeClass('on');
                                } else {
                                    $(this).prop("checked", checked).siblings('label').addClass('on');
                                }
                            }
                        }
                    });
                //}
                
                if(typeof callback == "function"){
                    callback.call(this);
                }
            };

            // 초기 전체 체크박스 체크 여부에 따른 처리
            if ($(this).is(':checked')) {
                _checked();
            }
            // 전체 체크박스 클릭시
            $(this).on("click", function(e) {
                _checked();
            });
            
            //for(i = 0 ; i < name.length; i++) {
                //$(":input[name="+name[i]+"]")
                $(selector).on('click', function() {
                    if ($(this).attr("name") != _this.attr("name") && !$(this).is(':checked')) {
                        _this.prop('checked', false).siblings('label').removeClass('on');
                    }
                });
           //}
          
        },
        
        /**
         * 리스트 선택시 상세화면 show/hide 처리 
         * 클릭 대상에 listDetail 클래스 추가 
         * 예)
         * $('.listDetail').hcListDetail();
         */
        hcListDetail : 
        function () {
            $(this).on("click", function (e) {
                var target = $(this).parents('li');
                if ($(target).hasClass('on')) {
                    $(target).removeClass('on');   
                } else {
                    $(target).addClass('on');   
                }
                e.preventDefault();
            });
        },
        /**
         * 입력 태그에 숫자만 입력 가능 
         * 입력 대상에 number-format 클래스 추가 
         * option = true 의 경우 천단위 (,) 처리 
         * 예) 
         * $('.number-format').hcInputNumberFormat();
         */
        hcInputNumberFormat : 
        function (option) {
            $(this).on("click", function(event) {
                var value = $(this).val();
                if (value != null && value != '') {
                    if(value.indexOf(",") != -1) {
                        $(this).val(value.replace(/\,/g, ''));
                    }
                }
            }).on('keydown keyup', function(event){
                var value = $(this).val();
                if (value != null && value != '' && value != 0) {
                    value = value.replace(/[^0-9]/g, '');
                }
                
                $(this).val(value);
            }).on("blur" , function(){
                var value = $(this).val();
                if (value != null && value != '' && value != 0) {
                    value = value.replace(/[^0-9]/g, '');
                }
                if(option) {
                    value = $.hc.getMoneyFormat(value);
                }
                $(this).val(value);
            });
        },
        /**
         * 입력 태그에 숫자만 입력 가능 
         * 입력 대상에 number-format 클래스 추가 
         * option = true 의 경우 천단위 (,) 처리 
         * 예) 
         * $('.number-format').hcInputNumber();
         */
        hcInputNumber : 
        function (option) {
            $(this).on("click", function(event) {
                if (this.value && this.value.indexOf(",") != -1) {
                    this.value = this.value.replace(/\,/g, '');
                }
            }).on('keydown keyup', function(event){
                if (this.value) {
                    this.value = this.value.replace(/[^0-9]/g, '');
                }
            }).on("blur" , function(){
                var value = this.value;
                if (value) {
                    value = value.replace(/[^0-9]/g, '');
                }
                if(option) {
                    value = $.hc.getMoneyFormat(value);
                }
                this.value = value;
            });
        },
       
        // 
        /**
         * currency format
         * 금액 형식으로 값을 변경하여 보여준다.
         * 예)$(selector).hcCurrencyFormat();
         * 
         * @returns selector의 value를 화폐형식으로 변경.
         */
        hcCurrencyFormat : 
        function(){
            var tagName = $(this).prop("tagName"); 
            var value = (tagName == "INPUT")? $.trim($(this).val()) : $.trim($(this).html());
            value = parseInt(value, 10);
            value = value + "";
            
            var reg = /(^[+-]?\d+)(\d{3})/;
            while (reg.test(value)) {
                value = value.replace(reg, '$1' + ',' + '$2');
            }
            (tagName == "INPUT")? $(this).val(value) : $(this).html(value);
            // return value;
        },
        
        /**
         * value 또는 html의 화폐형식 값을 number 형식으로 리턴
         * 예)
         * $(selector).hcNumberFormat()
         * @returns $(selector) 
         */
        hcNumberFormat : 
        function(){
            var tagName = $(this).prop("tagName"); 
            var value = (tagName == "INPUT")? $.trim($(this).val()) : $.trim($(this).html());
            value = value.split(",").join("");
            return parseInt(value);
        },
        
        
        /**
         * handlebars 사용을 좀 더 확장하기 위한 utility
         * 예)
         * $(selector).hcHandlebars($("#handlebars-template"), mData);
         * 
         * @param template
         *            handlebars 템플릿 객체
         * @param data
         *            템플릿에 바인딩 될 데이터 셋 
         * @param type (optional)  
         *            append - 템플릿 영역을 계속 붙인다.
         *            reload - 템플릿 영역이 새로 만들어진다.
         */
        hcHandlebars :  
        function(args) {
            // function(template, data, type) {
            var compiled = {};
            var template;
            var data;
            var load;
            var initFlag = false;
            if(arguments.length > 1) {
                template = arguments[0];
                data = arguments[1];
                load = arguments[2];
            } else {
                var defaults = {
                        template:'',
                        data : {},
                        load: "reload",
                        convertOpt: null
                };
                var config = $.extend({}, defaults, args);
                
                template = config.template;
                data = config.data;
                load = config.load;
                if(config.convertOpt) {
                    window[template.attr("id")] = config.convertOpt;
                    initFlag = true;
                }
            }
            
            if (template instanceof jQuery) {
                template = $(template).html();
            }
            compiled[template] = Handlebars.compile(template);
            if(load == "append") {
                $(this).append(compiled[template](data));
            }else {
                $(this).html(compiled[template](data));
            }
            var tid = setInterval(function(){
                if($("#endHandlebars").length > 0)
                    clearInterval(tid);
            }, 100);
        },

      
        
        /**
         * 숫자 또는 화폐 형식의 문자열을 각각에 해당하는 이미지로 변경하여 출력한다.
         * 예)
         * $("#nums").hcImgNumber("-12,345.00");
         * 
         * @param formatedNumber  이미지 
         */
        hcImgNumber : 
        function(formatedNumber) {
            var imgNum = "";
            
            formatedNumber += "";
            
            if(typeof formatedNumber == "undefined" || formatedNumber == "" ) {
                this.after(imgNum);
                return;
            }

            $.each(this.nextAll(), function(i, obj){
                if($(this).is("span.hcnum")){
                    $(this).remove();
                }
            });
            for(i = 0 ; i <formatedNumber.length; i++) { 
                // 숫자 이면...
                if($.isNumeric(formatedNumber.charAt(i))){
                    imgNum += "<SPAN class='num"+formatedNumber.charAt(i)+" hcnum'></SPAN>";
                }else{
                    switch(formatedNumber.charAt(i)){
                        case "," : 
                            imgNum += "<SPAN class='comma hcnum'></SPAN>";
                            break;
                        case "." :
                            imgNum += "<SPAN class='dot hcnum'></SPAN>";
                            break;
                        case "-" :
                            imgNum += "<SPAN class='dash hcnum'></SPAN>";
                            break;
                        default :
                            imgNum += formatedNumber.charAt(i);
                            break;
                    }
                }
            }
            this.after(imgNum);
        },
        
        /**
         * 해당 화면 리스트의 총 개수
         * 예)
         * $.hc.hcListTotalCount();
         */
        hcListTotalCount : 
        function () {
            return this.children('li').length;
        },
        /**
         * 해당 화면 리스트의 금액의 합
         * $.hc.hcListTotalSum();
         */
        hcListTotalSum : 
        function () {
            var targetObj = this.children().siblings();
            var totalsum = 0;
            $.each(targetObj, function(idx, obj){
                var value = parseInt($(obj).find('.value strong').text().replace(',',''));
                totalsum += value;
            });
            return totalsum;
        },
        
        /**
         * 이용기간 검색시 기간 표시 처리 
         * 예) 
         * $("#searchDateTerm").hcSetSearchDateTerm({  
         *                        autoSet:true, // 처음 로드시 디폴트로 설정된 기간 체크 여부
         *                        selectRadio:"use_day", // 이용기간 라디오 name 
         *                        startInput:"searchDateStart", //이용기간 시작 입력폼 
         *                        endInput:"searchDateEnd" // 이용기간 끝 입력폼 
         *                        format:"yy-MM-dd" //이용기간 표시 포맷 });
         *                        
         * 이용기간 :<input type="radio" value="3" name="use_day" checked=checked/>3개월
         *         <input type="radio" value="6" name="use_day"/>6개월 
         * <input type="text" id="searchDateStart"/> ~ <input type="text" id="searchDateEnd"/>
         *                         
         * 
         * @param o 사용자 지정 변수
         */
        hcSetSearchDateTerm : 
        function (o) {
            var options = {
                autoSet:true,                   // 초기 로딩시
                selectRadio:"",                 // 선택된 라디오 데이터 name
                startInput:"searchDateStart",   // 입력된 시작 날짜 id
                endInput:"searchDateEnd",       // 입력된 끝 날짜 id
                format:"yyyy-MM-dd"             // 표시 날짜 포맷
            };
            var  dateTerms = {
                week: -7,
                one_month: "-1",
                two_month: "-2",
                three_month: "-3",
                six_month: "-6"
            }; 
            var dateObj = new Date();

            var config = $.extend({}, options, o);
            // 라디오로 선택 되었을 시의 기간 값
            var radioValue = $(":input[name=" + config.selectRadio + "]:checked").val();
            var radioStartInputVal = $.hc.date.addDate(dateObj, dateTerms[radioValue]);
            var radioEndInputVal = $.hc.date.currDate(dateObj);
            
            // 사용자가 입력 했을 시의 기간 값
            var userSetStartVal = $("#"+config.startInput).val(); 
            var userSetEndVal = $("#"+config.endInput).val(); 
            
            var returnValue = '';
            if (!config.autoSet &&
                    (radioStartInputVal != userSetStartVal
                            || radioEndInputVal != userSetEndVal)) {
                
                returnValue = $.hc.convertDateString(userSetStartVal, 'yyyyMMdd', config.format) 
                            + " ~ " 
                            + $.hc.convertDateString(userSetEndVal, 'yyyyMMdd', config.format);
               
            } else {
                switch (radioValue) {
                    case "today" :
                        returnValue = '1일';
                        break; 
                    case "week":
                        returnValue = '1주일';
                        break;
                    case "one_month":
                        returnValue = '1개월';
                        break;
                    case "two_month":
                        returnValue = '2개월';
                        break;
                    case "three_month":
                        returnValue = '3개월';
                        break;
                    case "six_month":
                        returnValue = '6개월';
                        break;
                }
            }
            
            $(this).text(returnValue);
        },

        /**
         * 접근성 문제로 pc화면에서는 select box를 사용하지 않기 때문에 
         * 동적으로 select box를 만들어야 하는 경우 개개인이 만들기에는 너무 복잡하기 때문에
         * select box를 만들기 위해서 제공되는 메소드
         *  
         * 예) $(selector).hcMakeCombo(option)
         * 
         * @param option
         *            object
         * 
         * name : radio button name 
         * text : 화면에 보여줄 text를 담은 object key (지정되지 않으면 text) 
         * value : radio button에 value 값을 담은 object key (지정되지 않으면 value)
         * data : data set (Object in Array) {data:[text:"3개월", value:"3"],[]....}
         * selected : request에서 넘어온 selected 값 (optional) 
         * unit : 센스리더용 Text (optional) 
         * onLoadAfter : execute when this load (optional)
         * onChangeAfter : change event callback (optional)
         * valid : need to check valid (optional)
         * 
         */
        hcMakeCombo :
        function(option) {
            // name, select, text, start, end, callback
            $(this).find(".list-option").children().not(":first").remove();
            
            var data = arguments[0];
            
            var config = {
                    name : "",
                    text:"text",
                    value:"value",
                    data:"",
                    selected:"",
                    unit:null,
                    onLoadAfter:"",
                    onChangeAfter:"",
                    valid:false
                    
            };
            config = $.extend(true, {}, config, option);

            if(config.name == "") config.name = $(this).find(":input[type=radio]").attr("name");
            if(config.unit == null || config.unit == "") config.unit = $(this).find("label").find("span").first().text();
            
            function setOptions(name, idx, text, value, unit, selected) {
                var date = new Date();
                
                var div = $("<div></div>");
                var label   = $("<label></lable>").attr("for", "_"+name+idx);
                var span    = $("<span></span>").addClass("hd-t");
                var input   = $("<input></input>").attr("type", "radio");
                
                span.text(unit);
                                
                input.attr("name", name);
                input.attr("id", "_"+name+idx);
                input.val(value);
                
                if(value == selected) {
                    label.addClass("on");
                    input.attr("readOnly", "readOnly");
                }
                if(idx==0 && config.valid) {
                    input.attr("valid","{\"required\":\"true\"}");
                }
                
                label.text(text).prepend(span);// .text(text);
                div.append(label).append(input);
                
                return div;
            }
            
            for(i = 0; i<config.data.length; i++ ) {
                var dom = null;
                dom = setOptions(config.name, i, config.data[i][config.text], config.data[i][config.value], config.unit, config.selected);
                $(this).find(".list-option").append(dom);
            }
            
            if(config.onChangeAfter != "") {
                $("input[name="+config.name+"]").on("change", function(ev){
                   config.onChangeAfter.call(this, $(this) , $(this).val());
                });
            }
                      
            $(this).selectRadio();
            if(config.onLoadAfter != "") {
                config.onLoadAfter.call(this);
            }
        },
        
        /**
         * 공인 인증 사용시 인증에 필요한 서명값 hidden 필드를 지정된 폼에 생성
         * 
         * @param aResult -
         *            서명값
         */
        hcMakeXecureWebField : function (aResult) {
            this.find('#signed_msg, #vid_msg ').remove();
            // SignVerifier 에서 사용할 msg
            $('<input>').attr({
                type: 'hidden',
                id: 'signed_msg',
                name: 'signed_msg',
                value: aResult
            }).appendTo(this);
            
            // VidVerifier 에서 사용할 msg
            var vid_msg = send_vid_info();
            $('<input>').attr({
                type: 'hidden',
                id: 'vid_msg',
                name: 'vid_msg',
                value: vid_msg
            }).appendTo(this);
        },
        hcMakeXecureWebFieldMobile : function () {
            this.find('#signed_msg, #vid_msg, #pid ').remove();
            
            // SignVerifier 에서 사용할 msg
            $('<input>').attr({
                type: 'hidden',
                id: 'signed_msg',
                name: 'signed_msg',
                value: ''
            }).appendTo(this);
            
            // VidVerifier 에서 사용할 msg
            $('<input>').attr({
                type: 'hidden',
                id: 'vid_msg',
                name: 'vid_msg',
                value: ''
            }).appendTo(this);
            
            // VidVerifier 에서 사용할 msg
            $('<input>').attr({
                type: 'hidden',
                id: 'pid',
                name: 'pid',
                value: ''
            }).appendTo(this);
        },
        /**
         * 이중 서브밋 방지용 스크립트. 사용하지 않음.
         * 예) $(selector).hcSubmit();
         * @returns 
         */
        hcSubmit : function() {
            var $form = $(this);

            if ($form.data('submitted') === true) {
              // Previously submitted - don't submit again
              return this;
            } else {
              // Mark it so that the next submit can be ignored
                $form.data('submitted', true);
                $form.submit();
                
            }
        },
       
        /**
         * jquery on 메소드 기능 확장.
         * on메소드를 하나의 객체에 여러번 바인딩 하게 되면 
         * 한 번의 액션에도 여러 번 이벤트가 호출되는 현상을 줄이기 위해
         * 메소드 내부에서 off 메소드를 호출하도록 하였으나 
         * 다른 이벤트가 사라지는 현상이 있어 off 메소드를 사용하지 않음
         * 예) $(selector).hcOn(event, callback);
         * @returns 
         */
        hcOn : function(event, callback) {
            //$(this).off(event, callback);
            $(this).on(event, callback);
        }
    });
})(jQuery);


