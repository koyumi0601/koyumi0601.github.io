(function($){
    var chartHostType;
    var chartData;
    var SESS_VAL;
    $.extend({
        hc:{
            // 입력한 숫자를 지정한 자리수 만큼 맞추기 위해 숫자 앞에 0을 붙인다.
            /**
             * Statements $.hc.zeroFill(number, width)
             * @param number : 자리수를 맞추고 싶은 숫자
             * @param width : 총 자리수
             * @returns 총 자리수에 맞춰진 숫자
             */
            zeroFill:
                function( number, width ) {
                    width -= number.toString().length;
                    if ( width > 0 ) {
                        return new Array( width + (/\./.test( number ) ? 2 : 1) ).join( '0' ) + number;
                    }
                    return number + ""; // always return a string
            },
            // 숫자를 화폐 형식으로 변경
            /**
             *  $.hc.currencyFormat(number);
             * @param value 일반적인 숫자
             * @returns 화폐형식의 포맷
             */
            currencyFormat :
                function(value){
                    value = parseInt(value, 10);
                    value = value + "";

                    var reg = /(^[+-]?\d+)(\d{3})/;
                    while (reg.test(value)) {
                        value = value.replace(reg, '$1' + ',' + '$2');
                    }
                    return value;
            }, ivl : function(strVal) {
                if ( strVal == null ) return 0;
                if ( $.trim(strVal).length <= 0 ) return "0";
                return parseInt( strVal,10 );
            },
            //입력한 키를 사용하여 메세지 파일로부터 특정 문자열을 불러온다.
            /**
             * setProperties는 자신이 사용할 경로/파일을 지정한다.
             * $.hc.messages.setProperties({options})
             * options.path = 메시지 파일 경로
             * options.name = 메시지 파일 이름
             * options.mode = 메시지 출력 방식
             * USAGE : $.hc.messages.setProperties({name:"my-messages"});
             *
             * key = message_key
             * key는 {namespace}.적용대상.적용내용 으로 만든다.
             * {namespace}로 key의 범위를 한정할 수 있다.
             * 두 단어 이상으로 설정해야 명확히 구분할 수 있다. (하단의 주의 참조)
             * 예) validation Email 포맷 체크 메시지의 경우
             * hc.valid.email.format=Email 형식이 올바르지 않습니다.
             *  {hc.valid} namespace, email 적용대상, format 적용내용
             *
             * vars = 메시지에 동적으로 할당될 변수
             * $.hc.messages.get(key, vars1, vars2, ...... varsN) 또는 $.hc.messages.get({key:"key", vars:"vars1, vars2, ....varsN"})
             *USAGE :
             * $.hc.messages.get("alert_end_loading", "이름", "나이", location); 또는
             * $.hc.messages.get({key:"alert_end_loading", vars:"이름, 나이, 지역"});
             *
             * 주의 : 한 페이지에 setProperties를 여러번 호출할 경우 (즉, 여러 개의 message 파일 로드)
             * map형식([key]=value) 으로 메시지를 담고 있기 때문에 key가 겹치면 원하지 않는 메시지가 출력될 수 있습니다.
             */
            messages:
            {
                msg : '',  // 함수 내부에서 사용할 메시지 객체를 담을 변수

                defaults : {
                    path: '/docfiles/resources/pc/js/properties/',
                    name:'validation-messages',
                    mode:"map",
                    callback:""
                },

                 /**
                 *  $.hc.messages.setProperties({options})
                 * @param options :
                 *  path - properties path,
                 *  name - properties file name,
                 *  mode: map,
                 *  callback : function
                 */
                setProperties:
                    function(options){
                        var self = this;
                        self.config = $.extend({},self.defaults, options);
                        $.i18n.properties(self.config);
                        self.msg = $.extend({}, self.msg, $.i18n);
                },

                 /**
                 * 
                 * $.hc.messages.get(propKey, param1, param2, .... paramN)
                 * $.hc.messages.get({key : propKey, vars : param1, param2, .... paramN})
                 * @param o :
                 * properties map key,
                 * parameters
                 * @returns {String} : map key 로 불려진 메시지에 parameter 값들이 매핑된 텍스트
                 */
                get :
                    function(o) {
                        var self = this;
                        var msg = "";

                        if(self.msg == '') {
                            self.setProperties();
                        }

                        if(typeof o == "undefined"){
                            alert("Message key값을 입력해 주세요.");
                            return;
                        } else {
                            if(typeof o == "string" && arguments.length == 1) {
                                msg = self.msg.prop(o);
                            }else if(arguments.length > 1) {
                                var key = arguments[0];
                                var vars = "";
                                for(i = 1; i < arguments.length; i++){
                                    vars += (vars == "")? arguments[i] : ","+arguments[i];
                                }
                                msg = self._get(self.msg.prop(key), vars);
                            }else if(typeof o == "object") {
                                if(o.hasOwnProperty("vars")) {
                                    msg = self._get(self.msg.prop(o.key), o.vars);
                                } else {
                                    msg = self.msg.prop(o.key);
                                }
                            }
                        }

                        return msg;
                },

                /**
                 *  $.hc.messages.getPropertiesMap(namespace)
                 * @param namespace :
                 * message 를 구분 하기 위한 이름.
                 * 예를 들어 validation 에 관련된 메시지라면 키 이름에 valid 를 namespace로 선언
                 * hc.valid.email, hc.valid.required, hc.valid.digit...
                 * @returns {Array} : namespace가 포함된 메시지 map
                 */
                getPropertiesMap:
                    function(namespace){
                        var self = this;
                        var msg = "";
                        var data = [];

                        $.each(self.msg.map, function(key, mData){
                            if(typeof namespace == "undefined" || namespace == "" ){
                                data[key] = mData;
                            }else if(key.indexOf(namespace) >= 0) {
                                data[key] = mData;
                            }
                        });
                        return data;
                },

                /**
                 *  $.hc.messages.getPropertiesKeys(namespace)
                 * @param namespace
                 * message 를 구분 하기 위한 이름.
                 * 예를 들어 validation 에 관련된 메시지라면 키 이름에 valid 를 namespace로 선언
                 * valid.email, valid.required, valid.digit...
                 * @returns {Array} : namespace가 포함된 메시지 key
                 */
                getPropertiesKeys:
                    function(namespace){
                        var self = this;
                        var msg = "";
                        var data = [], i = 0;

                        $.each(self.msg.map, function(key, mData){
                            if(typeof namespace == "undefined" || namespace == "" ){
                                data[i++] = key;
                            }else if(key.indexOf(namespace) >= 0) {
                                data[i++] = key;
                            }
                        });
                        return data;
                },
                /**
                 *  this._get(message, param1, param2,...paramN)
                 * @param message : properties file에서 불러온 메시지
                 * @param vars : 메지시에 치환될 변수 값
                 * @returns : 변수가 치환된 완전한 메시지
                 */
                _get:
                    function(message, vars) {
                        var temp = [];
                        var t_var = [];
                        var temp = message.split(/(\{\d+\})/g);
                        if(typeof vars != "undefined"){
                            if(vars.indexOf(",") > 0) {
                                t_var = vars.split(",");
                            }else{
                                t_var[0] = vars;
                            }
                        }
                        for(var i = 0, j = 0; i < temp.length; i++ ){
                            if((/(\{\d+\})/).test(temp[i]) && vars.length > j){
                                temp[i] = $.trim(t_var[j++]);
                            }
                        }
                        return temp.join("");
                },

                clearMessage :
                    function() {
                        self.setProperties();
                }

            },
            /**
             *  $.hc.convertDateString(value, fromPattern, toPattern)
             * 날짜를 지정한 포맷 형식으로 반환
             * @param value - 변환대상 날짜
             * @param fromPattern- 현재 패턴
             * @param toPattern- 변환 패턴
             * @returns
             */
            convertDateString :
            function(value, fromPattern, toPattern) {
                if (typeof value == 'undefined' || value == null || value == '')
                    return value;

                var date = $.format.date(value, fromPattern);
                return $.format.date(date, toPattern);
            },
            /**
             *  $.hc.getMoneyFormat("3230000") // 3,230,000
             * @param strVal - 변환 대상 금액
             * @returns #,###.##
             *
             * 천단위 , 삽입
             */
            getMoneyFormat :
            function(strVal) {
                strVal = strVal + "";
                if (typeof strVal == 'undefined' || strVal == null || strVal == '')
                    return "0";
                if(strVal.indexOf(",") != -1) {
                    strVal = strVal.replace(/,/g,"");
                }
                if(!$.isNumeric(strVal))
                    return "0";

                return $.format.number(parseFloat(strVal), '#,###.##');
            },


            /**
             *  
             * @param strVal2 - 변환 대상 금액
             * @returns #,###.9
             *
             * 천단위 , 삽입
             */
            getMoneyFormat1 :
            function(strVal) {
                strVal = strVal + "";
                if (typeof strVal == 'undefined' || strVal == null || strVal == '')
                    return "0";
                if(strVal.indexOf(",") != -1) {
                    strVal = strVal.replace(/,/g,"");
                }
                if(!$.isNumeric(strVal))
                    return "0";

                return $.format.number(parseFloat(strVal), '#,###.9');
            },



            /**
             *  
             * @param strVal2 - 변환 대상 금액
             * @returns #,###.99
             *
             * 천단위 , 삽입
             */
            getMoneyFormat2 :
            function(strVal) {
                strVal = strVal + "";
                if (typeof strVal == 'undefined' || strVal == null || strVal == '')
                    return "0";
                if(strVal.indexOf(",") != -1) {
                    strVal = strVal.replace(/,/g,"");
                }
                if(!$.isNumeric(strVal))
                    return "0";

                return $.format.number(parseFloat(strVal), '#,###.99');
            },


            /**
             *  
             * @param  strVal  변환 대상 값
             * @returns #,###.999
             *
             * 소숫점 3자리 까지 표시
             */
            getDecimalFormat3 :
            function(strVal) {
                strVal = strVal + "";
                if (typeof strVal == 'undefined' || strVal == null || strVal == '')
                    return "0";
                if(strVal.indexOf(",") != -1) {
                    strVal = strVal.replace(/,/g,"");
                }
                if(!$.isNumeric(strVal))
                    return "0";

                return $.format.number(parseFloat(strVal), '#,###.999');
            },



            // substring
            substring :
            function (value, from, to) {
                return value.substring(from, to);
            },
            /**
             * 리스트의 총합계 영역 데이터 셋팅
             * 사용) $.hc.hcSetTotalInfo('grid_list', 'box-summary2-total');
             * @param targetListClass - 리스트를 감싸고 있는 div class 명
             * @param targetTotalClass - 총 합계를 나타내는 영역의 class 명
             */
            setTotalInfo :
            function (targetListClass, targetTotalClass) {
                var totalCount = $('.'+targetListClass).hcListTotalCount();
                var totalSum = $('.'+targetListClass).hcListTotalSum();
                $('.'+targetTotalClass).find('.num em').text($.hc.getMoneyFormat(totalCount));
                $('.'+targetTotalClass).find('.price strong').text($.hc.getMoneyFormat(totalSum));

            },
            /**
             * 라디오버튼으로 기간설정 공통 메소드 내부에서 호출
             */
            date:{

                 /**
                 * 
                 * $.hc.date.currDate(new Date());  // 20130521
                 * @param dateObj (데이터 객체)
                 * @returns yyyyMMdd
                 *
                 * 현재 날짜를 yyyyMMdd 포맷으로 리턴
                 */
                currDate :
                function(dateObj)  {
                    var year = dateObj.getFullYear();
                    var month = $.hc.zeroFill(dateObj.getMonth()+1, 2);
                    var day = $.hc.zeroFill(dateObj.getDate(), 2);
                    return year+month+day;
                },

                /**
                 * 
                 * $.hc.date.addDate('20130403', -2); // 20130401
                 * $.hc.date.addDate('20130403', "-2"); // 20130203
                 *
                 * @param currDate 특정일
                 * @param days Number 인 경우 - 날짜로 계산, String인 경우 월로 계산
                 * @returns yyyyMMdd
                 *
                 * 특정일로 부터 days 만큼의 날을 더하거나 뺀 결과를 리턴한다.
                 */
                addDate :
                function(currDate, days) {
                    var resultDate = null;
                    var diffMonth = null;
                    var inputDay = null;
                    var lastDay = null;
                    if(typeof days == "number") {
                        resultDate = new Date(Date.parse(currDate) + days*1000*60*60*24);
                    }else {
                        diffMonth = parseInt(currDate.getMonth()) + parseInt(days);

                        inputDay = currDate.getDate();
                        lastDay = (new Date(currDate.getFullYear(), diffMonth, 0)).getDate();

                        currDate.setDate(Math.min(inputDay, lastDay));
                        currDate.setMonth(diffMonth);
                        resultDate = currDate;
                    }

                    var year = resultDate.getFullYear();
                    var month = $.hc.zeroFill(resultDate.getMonth()+1, 2);
                    var day = $.hc.zeroFill(resultDate.getDate(), 2);
                    return year+month+day;
                }
            },

            /**
             * $.hc.convertFunc(key, item, context)
             * @param key handlebars context key
             * @param item handlebars convertOpt
             * @param context handlebars context
             * number : 화폐 형태의 포맷으로 변환
             * date : fromFormat 을 toFormat으로 변환  ( 'yyyyMMdd' -> 'yyyy년MM월dd일')
             * custom : func를 지정하여 함수 실행 값을 리턴
             * string : half (전각을 반각으로), full(반각을 전각으로)
             */
            convertFunc :
            function(key, item, context, temp) {
                var data;
                data = context[key];

                if(item.type == "number") {
                    data = $.hc.getMoneyFormat(data);
                }else if(item.type == "money") {
                    data = $.hc.getMoneyFormat(data);
                }else if(item.type == "date") {
                    if(typeof item.fromFormat == "undefined" || item.fromFormat == "") {
                        if(typeof item.toFormat != "undefined" && item.toFormat != "") {
                            data = $.hc.convertDateString(data, 'yyyyMMdd', item.toFormat );
                        }
                    } else {
                        data = $.hc.convertDateString(data, item.fromFormat, item.toFormat );
                    }
                } else if(item.type == "custom") {
                    if(item.func) {
                        data = item.func(key, item, context, temp);
                    } else {
                        console.log("custom function을 지정하여 주세요. func item 추가");
                    }
                } else if(item.type == "string") {
                    //반각으로 만들기
                    if(item.format== "half"){
                        data = $.hc.getConvertFullToHalf(data);
                    //전각으로 만들기
                    }else if(item.format == "full"){
                        data = $.hc.getConvertHalfToFull(data);
                    }
                }
                context[key] = data;
            },

            /**
             * $.hc.getConvertFullToHalf(x_char)
             * @param x_char 전각문자
             * 전각을 반각으로 변환하여 리턴한다.
             */
            getConvertFullToHalf:
            function(x_char) {
                var x_2byteChar = new String;
                var len = x_char.length;
                    for (var i = 0; i < len; i++) {
                        var c = x_char.charCodeAt(i);

                        if (c >= 65281 && c <= 65374 && c != 65340) {
                            x_2byteChar += String.fromCharCode(c - 65248);
                         } else if (c == 8217) {
                           x_2byteChar += String.fromCharCode(39);
                         } else if (c == 8221) {
                           x_2byteChar += String.fromCharCode(34);
                         } else if (c == 12288) {
                           x_2byteChar += String.fromCharCode(32);
                         } else if (c == 65507) {
                           x_2byteChar += String.fromCharCode(126);
                         } else if (c == 65509) {
                           x_2byteChar += String.fromCharCode(92);
                         } else {
                           x_2byteChar += x_char.charAt(i);
                         }
                   }
                 return x_2byteChar;
            },

            /**
             * $.hc.getConvertFullToHalf(x_char)
             * @param x_char 전각문자
             * 전각을 반각으로 변환하여 리턴한다.
             */
            getConvertHalfToFull :
            function(x_char) {
              var x_2byteChar = "";
              var c = x_char.charCodeAt(0);
              if(32 <= c && c <= 126) {
                  if(c == 32) {
                      x_2byteChar = unescape("%uFFFC");
                  } else {
                      x_2byteChar = unescape("%u"+(c+65248).toString(16));
                  }
              }
              return x_2byteChar;
            },


            transkeyClose:
            /**
             * Statements $.hc.transkeyClose(element, callback)
             * @param elem transkey가 적용된 element
             * @param func  transkey close custom callback
             *
             * transkey 키패드 Close callback - validataion 공통처리 관련 method
             */
            function(elem, func){
                /*elem.on("blur.jqueryValid", function(e){
                    console.log(func);
                    func(e);
                });*/
                if(elem.val().length == elem.prop("maxlength")){
                    $.hc.removeErrorMsg(elem);
                } else {
                    //elem.focus();
                    elem.blur();
                }
                //elem.removeAttr("ReadOnly");
            },

            /**
             * $.hc.removeErrorMsg(elem)
             * @param element
             *
             * transkeyClose 내부에서 호출 - validation 처리 관련 method
             */
            removeErrorMsg:
            function(element) {
                var self = this;

                //var fixed = self.fixed;
                //var config = self.config;

                var errorViewTarget = element.siblings();
                var removeFlg = true;

                $.each(errorViewTarget, function(idx, obj){
                    if ($(this).hasClass("validate")) {
                        $(this).removeClass("error").html('');
                        removeFlg = false;
                    }
                });
                if (removeFlg) {
                    errorViewTarget = element.parents('.form').find(".validate");
                    if (errorViewTarget.hasClass("validate")) {
                        errorViewTarget.removeClass("error").html('');
                        removeFlg = false;
                    }
                }
                element.removeClass("error");

                //fixed.firstErrorMsg = '';
            },

            /**
             * $.hc.createChart(obj, type, data);
             * @param obj document (this)
             * @param type chart type
             * @param data chart data (Array)
             * 
             * 차트 정의 - 퍼블리싱 완료후 변경 필요
             */
            createChart :
                function(obj, type, data) {
                // 차트
                chartData = data;
                chartHostType = type;
                var THIS = obj;
                var chartType = "";

                if (type=="vStackBar") {
                    chartType = "/js/chart/vStackBar.js";
                } else if (type=="vBar") {
                    chartType = "/js/chart/vBar.js";
                } else if (type=="hStackBar") {
                    chartType = "/js/chart/hStackBar.js";
                } else if (type=="hBar") {
                    chartType = "/js/chart/hBar.js";
                }
                IChart.setUp(["/js/chart/kinetic-v4.4.4.min.js", chartType, "/js/chart/IChartCanvas.js"], $.hc.chartStart, THIS, []);
                if (typeof callback == 'function') {
                    callback.call(this);
                }
            },
            
            /**
             * $.hc.getChartData();
             *  
             * 각 차트 host.js 에서 data 부르는 함수
             */
            getChartData :
                function() {
                return chartData;
            },

            //IChart.setUp이 끝나고 호출하는 callback method
            /**
             * $.hc.chartStart(type);
             *  
             * IChart.setUp이 끝나고 호출하는 callback method
             */
            chartStart :
                function(type) {
                var chartType = "";
                if (chartHostType=="vStackBar") {
                    chartType = "/js/chart/vStackBarHost.js";
                } else if (chartHostType=="vBar") {
                    chartType = "/js/chart/vBarHost.js";
                } else if (chartHostType=="hStackBar") {
                    chartType = "/js/chart/hStackBarHost.js";
                } else if (chartHostType=="hBar") {
                    chartType = "/js/chart/hBarHost.js";
                }
                IChart.util.loadScript(chartType);
            },
            /**
             * $.hc.handlbars({template:$(handlebarsTemplateElenetId), data:dataObject, load:loadType, condition:convertObject})
             * @param args
             * template $(handlebarsTemplateElenetId)
             * data dataObject
             * load loadType - reload, append (optional)
             * condition convertObject (optional)
             * @returns 완성된 html코드
             */
            handlebars :
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
                        ///console.log(config.convertOpt);
                        if(config.convertOpt) {
                            window[template.attr("id")] = config.convertOpt;
                            //console.log(window[template.attr("id")]);
                            initFlag = true;
                        }
                    }

                    if (template instanceof jQuery) {
                        template = $(template).html();
                    }
                    compiled[template] = Handlebars.compile(template);
                    return compiled[template](data);
            },

            /**
             * $.hc.callOfferBox(classname)
             * @param selector
             * class name 위치에 offerBox 레이어 팝업을 띄워준다.
             */
            callOfferBox :
                function(selector, mbflag) {
                var $obj = null;
                var url = "";
                if($("#"+selector).length > 0) {
                   $obj =  $("#"+selector);
                } else {
                   $obj =  $("."+selector);
                }

                if(mbflag) {
                    url = "/cpa/bf/pop_offerbox_m.html";
                }else {
                    url = "/cpa/bf/pop_offerbox.html";
                }
                $obj.modalCon({
                    callbackBefore:function(){
                        
                        
                        var flag = false;
                        $.hcAjax({
                            category:"data",
                            async:false,
                            url:"/cpa/bf/CPABF0101_04.hc",
                            success:function(data){
                                
                                s_omni.pageName= "P:TT:메인:모듈:OfferBox";
                                s_omni.t();
                                
                                flag = true;
                            },
                            error : function(code, message, data){
                                if(typeof data.error_code != "undefined") code = data.error_code;
                                if("HDCUSRINF000" == code){
                                    var loginPath = "/cpm/mb/CPMMB0101_01.hc";
                                    var isLoginPath = window.location.pathname.indexOf(loginPath) >= 0;
                                    location.href=loginPath;
                                    if(!isLoginPath) {

                                    }else{
                                    //    alert("로그인 후 이용가능한 서비스입니다.");
                                    }
                                }
                                else if("HDCUSRINF065" == code) {
                                    alert(message);
                                }
                            }
                        });
                        return flag;
                    },
                    url:url,
                    layerWrap:".layer-type2",
                    sCheck:true  // unescape 처리
                });
            }
            ,
            /**
             * $.hc.makeStoreNoFormat(str)
             * @param str : 가맹점번호에 '-'를 첨가한다.
             * @returns 포맷된 가맹점 번호
             */
            makeStoreNoFormat:function(str) {
                if ( str == null ) return "";
                var strText = $.trim(str); 
                var temp = "";
                if ( strText.length >= 9 ) {
                    temp = strText.substring( 0, 3 ) + "-" + strText.substring( 3 );
                }
                return temp;
            },
            /**
             * $.makeBizNoFormat(str)
             * @param str : 사업자번호에 '-'를 첨가한다.
             * @returns 포맷된 사업자번호
             */
            makeBizNoFormat:function(str) {
                if ( str == null ) return "";
                var temp = str;

                if ( str.length == 10 ) {
                    temp = str.substring( 0, 3 ) + "-" + str.substring( 3, 5 ) + "-" + str.substring( 5, 10 );
                }
                return temp;
            },
            /**
             * $.hc.makeStoreNoFormat(str)
             * @param str : 가맹점번호에 '-'를 첨가한다.
             * @returns 포맷된 가맹점 번호
             */
            getStoreCardNo:function(cardNo) {
                var DINERS_CARD_NO_LENGTH = 14;
                var NORMAL_CARD_NO_LENGTH = 16;
                var maskedCardNo = '';
                    if ( cardNo == null ) return "";
                    cardNo = $.trim(cardNo);
                    maskedCardNo = cardNo.substring( 0, 4 );
                    maskedCardNo = maskedCardNo+"-";
                    maskedCardNo = maskedCardNo+cardNo.substring( 4, 6 );

                    if ( cardNo.length == DINERS_CARD_NO_LENGTH ) {
                        maskedCardNo = maskedCardNo+"****-*" ;
                        maskedCardNo = maskedCardNo+cardNo.substring( 11 );
                    }
                    else if ( cardNo.length == NORMAL_CARD_NO_LENGTH ) {
                      maskedCardNo = maskedCardNo+"**-****-*";
                      maskedCardNo = maskedCardNo+cardNo.substring(13);
                    }
                    else{
                        for(var i =0; i<cardNo.length;i++){
                            maskedCardNo=maskedCardNo+"#";
                        }
                    }

                    return maskedCardNo;
            },
            

            /**
             * $.hc.playVideo(options)
             * @param options {videoId:{title:"",caption:""},{ .... }}
             * @returns 
             * 문화 공연 영역에 비디오 플레이를 담당한다.
             */
            playVideo : function(options) {

                var config = $.extend({},options);

                var deviceType = (config.deviceType != null)? config.deviceType : null;

                if(deviceType == "PC") {
                    $.hc.getPCVideo(config);
                }else if(deviceType == "TABLET") {
                    $.hc.getTabletVideo(config);
                }else if(deviceType == "MOBILE") {
                    $.hc.getMobileVideo(config);
                }else {
                    alert("[PlayVideo] : DEVICE TYPE을 넣어주세요\nex) option : {\"deviceType\":\"${deviceChannelStorage.deviceType}\", ");
                }

            },
            
            
            /**
             * $.hc.getPCVideo(config)
             * @param config 비디오 정보
             * @returns 
             * PC에서의 비디오 플레이를 담당한다.
             */
            getPCVideo : function(config) {

                var attributes = { id:"videoPlay" };
                var flashvars = { };
                var params = {allowScriptAccess:"always", wmode : "transparent"}; //2015-07-16 

                var o = $(".list-img1 > li").get(0);

                if($(".list-img1 > li").length <=0) return;

                var videoId = $(o).find("a").attr("href");
                var key = $(o).find("a").attr("seq");
                var title = $(o).find("a").prev().attr("alt");
                var caption = "";
                var pane = $('#playerSubtitle');
                pane.jScrollPane();
                var api = pane.data('jsp');


                $("#video_title").empty();
                $("#video_title").html(title);

                caption = ($("#"+key).length > 0)?  $("#"+key).val() : config[videoId].caption;

                api.getContentPane().html("<p>"+ caption + "</p>");
                api.reinitialise();

                if(navigator.appName.toLowerCase().indexOf("explorer") >= 0) { //I.E에서 flash를 이용한 비디오 플레이

                    $.ajax({
                        url: "/js/libs/swfobject.js",
                        dataType: "script",
                        success: function(data){
                            swfobject.embedSWF("http://www.youtube.com/v/"+videoId+"?version=3&amp;hl=ko_KR", "videoPlay", "410", "100%", "9.0.0", null, flashvars, params, attributes);
                        }
                    });


                    $(".list-img1 > li").find("a").on("click", function(e) {
                        e.preventDefault();
                        title = $(this).prev().attr("alt");
                        videoId = $(this).attr("href");
                        key = $(this).attr("seq");
                        swfobject.removeSWF("videoPlay");

                        $("#video_title").html(title);
                        //$("#playerSubtitle > p").html(config[videoId].caption);
                        $(".wrap-movie").prepend("<div class='area-player' id='videoPlay'></div>");
                        swfobject.embedSWF("http://www.youtube.com/v/"+videoId+"?version=3&amp;hl=ko_KR", "videoPlay", "410", "100%", "9.0.0", null, flashvars, params, attributes);

                        caption = ($("#"+key).length > 0)?  $("#"+key).val() : config[videoId].caption;

                        api.getContentPane().html("<p>"+ caption + "</p>");
                        api.reinitialise();
                    });
                } else {   // IE 이외의 브라우저에서 iframe을 이용한 비디오 플레이

                    $("#videoPlay").append("<iframe width='410' height='100%' src='http://www.youtube.com/v/"+videoId+"?version=3&amp;hl=ko_KR' frameborder='0' allowfullscreen allowScriptAccess></iframe>");

                    $(".list-img1 > li").find("a").on("click", function(e) {
                        e.preventDefault();
                        title = $(this).prev().attr("alt");
                        videoId = $(this).attr("href");
                        key = $(this).attr("seq");
                        $("#videoPlay").empty();
                        $("#video_title").html(title);
                        //$("#playerSubtitle > p").html(config[videoId].caption);
                        $("#videoPlay").append("<iframe width='410' height='100%' src='http://www.youtube.com/v/"+videoId+"?version=3&amp;hl=ko_KR' frameborder='0' allowfullscreen allowScriptAccess></iframe>");

                        caption = ($("#"+key).length > 0)?  $("#"+key).val() : config[videoId].caption;
                        api.getContentPane().html("<p>"+ caption + "</p>");
                        api.reinitialise();
                    });
                }
            },

            /**
             * $.hc.getTabletVideo(config)
             * @param config 비디오 정보
             * @returns 
             * Tablet 에서의 비디오 플레이를 담당한다. 
             * 
             */
            getTabletVideo : function(config){
                var o = $(".list-img1 > li").get(0);

                if($(".list-img1 > li").length <=0) return;

                var videoId = $(o).find("a").attr("href");
                var key = $(o).find("a").attr("seq");
                var title = $(o).find("a").prev().attr("alt");

                var pane = $('#playerSubtitle');
                pane.jScrollPane();
                var api = pane.data('jsp');


                $("#video_title").empty();
                $("#video_title").html(title);

                var caption = ($("#"+key).length > 0)?  $("#"+key).val() : config[videoId].caption;
                api.getContentPane().html("<p>"+ caption + "</p>");
                api.reinitialise();

                $("#videoPlay").html("");

                $("#videoPlay").append("<video id='player1' class='youtube-player' width='100%' height='270'><source type='video/youtube' src='http://www.youtube.com/watch?v="+videoId+"'></video>");

                var player = new MediaElementPlayer('#player1');

                $(".list-img1 > li").find("a").on("click", function(e) {
                    e.preventDefault();
                    title = $(this).prev().attr("alt");
                    videoId = $(this).attr("href");
                    key = $(this).attr("seq");
                    $("#videoPlay").empty();
                    $("#video_title").html(title);

                    $("#videoPlay").html("");

                    $("#videoPlay").append("<video id='player1' class='youtube-player' width='100%' height='270'><source type='video/youtube' src='http://www.youtube.com/watch?v="+videoId+"'></video>");

                    var player = new MediaElementPlayer('#player1');

                    var caption = ($("#"+key).length > 0)?  $("#"+key).val() : config[videoId].caption;
                    api.getContentPane().html("<p>"+ caption + "</p>");
                    api.reinitialise();
                });
            },
            /**
             * $.hc.getMobileVideo(config)
             * @param config 비디오 정보
             * @returns 
             * Mobile 에서의 비디오 플레이를 담당한다. 
             * 
             */
            getMobileVideo : function(){

                var o = $(".list-img1 > li").get(0);

                if($(".list-img1 > li").length <=0) return;

                var videoId = $(o).find("a").attr("href");
                var title = $(o).find("a").prev().attr("alt");

                $("#video_title").html(title);

                $("#videoPlay").html("");

                $("#videoPlay").append("<video id='player1' class='youtube-player' width='100%' height='270'><source type='video/youtube' src='http://www.youtube.com/watch?v="+videoId+"'></video>");

                var player = new MediaElementPlayer('#player1');

                $(".list-img1 > li").find("a").on("click", function(e) {
                    e.preventDefault();
                    title = $(this).prev().attr("alt");
                    videoId = $(this).attr("href");
                    $("#videoPlay").empty();
                    $("#video_title").html(title);

                    $("#videoPlay").html("");

                    $("#videoPlay").append("<video id='player1' class='youtube-player' width='100%' height='270'><source type='video/youtube' src='http://www.youtube.com/watch?v="+videoId+"'></video>");

                    var player = new MediaElementPlayer('#player1');
                });
            },

            /**
             * 전자 스페이스를 trim해서 반환한다.<br>
             *
             * @param   String str :
             * @return  String
             */
            getSealTrim : function(strVal){
                var str = "";

                for(var i=0; i<strVal.length; i++){
                    if(strVal.substring(i,i+1) != '　'){
                        str += strVal.substring(i,i+1);
                    }
                }

                /*
                str = strVal.replace( '　', ' ' );
                 */

                str = $.trim(str);
                return str;
            },

            /**
             * 계좌번호를 마스킹 처리한다.
             * 앞 세자리와 마지막 한자리를 제외한 나머지는 모두 *처리
             * (입력된 계좌번호의 길이 == 마스킹 처리된 계좌번호의 길이)
             *
             * @param accountNo 마스킹 처리할 계좌번호 1234567890
             * @return 마스킹 처리된 계좌번호 123******0,
             *         계좌번호가 null이면 empty string,
             *         5자리 미만인 경우 입력된 계좌번호를 모두 #으로 변환한 값
             */
            getAccountNo : function(accountNo){

                var REG_EXP = "[0-9]";
                var ERR_STRING = "#";

                if ( accountNo == null || accountNo == "") return "";

                accountNo = $.trim(accountNo);
                //앞세자리와 마지막 한자리를 보여주기 위해서는 적어도 5자리이상이 되어야함.
                if ( accountNo.length < 5 ) return accountNo.replaceAll( REG_EXP, ERR_STRING );

                var maskedAccountNo = "";

                maskedAccountNo += accountNo.substring( 0, 3 );
                for (var i = 0; i < accountNo.length - 4; i++ ){
                    maskedAccountNo += "*";
                }
                maskedAccountNo +=  accountNo.substring( accountNo.length - 1 );

                return maskedAccountNo;
            },

            /**
             * 문자열을 실수(float)로 변환한다.<br>
             *
             * @param String val 변환대상 문자열
             * @return float
             */
            getFvl :
            function(strVal) {

                if (typeof strVal == 'undefined' || strVal == null || strVal == '')
                    return 0;
                if(!$.isNumeric(strVal))
                    return 0;

                return $.format.number(parseFloat(strVal));
            },

            /**
             * 카드번호를 마스킹 처리한다.
             * 다이너스의 경우(14자리) : #***-******-**##
             * 로컬, 비자, 마스터의 경우(16자리) : #***-****-****-###*
             *
             * @param cardNo 마스킹 처리할 카드번호
             * @return 마스킹 처리된 카드번호,
             *         카드번호가 null이면 empty string,
             *         카드번호 길이가 14혹은 16이 아닌경우 입력된 카드번호를 모두 #으로 변경한 값
             */
            getCardNo:function(cardNo){
                var DINERS_CARD_NO_LENGTH = 14;
                var NORMAL_CARD_NO_LENGTH = 16;
                var CARD_NO_LENGTH_10 = 10;
                var CARD_NO_LENGTH_12 = 12;
                var CARD_NO_LENGTH_15 = 15;
                var maskedCardNo
                
                if ( cardNo == null ) return "";
                cardNo = $.trim(cardNo);
                
                maskedCardNo = cardNo.substring( 0, 1 );

                if ( cardNo.length == DINERS_CARD_NO_LENGTH ) {
                    maskedCardNo = maskedCardNo + "***-******-**" ;
                    maskedCardNo = maskedCardNo + cardNo.substring( 12 );
                }
                else if ( cardNo.length == NORMAL_CARD_NO_LENGTH ) {
                    maskedCardNo = maskedCardNo + "***-****-****-";
                    maskedCardNo = maskedCardNo + cardNo.substring( 12, 15 );
                    maskedCardNo = maskedCardNo + "*";
                }
                else if ( cardNo.length == CARD_NO_LENGTH_10 ) {
                    maskedCardNo = maskedCardNo + "***-****-" ;
                    maskedCardNo = maskedCardNo + cardNo.substring( 8 );
                }
                else if ( cardNo.length == CARD_NO_LENGTH_12 ) {
                    maskedCardNo = maskedCardNo + "***-****-**" ;
                    maskedCardNo = maskedCardNo + cardNo.substring( 10 );
                }
                else if ( cardNo.length == CARD_NO_LENGTH_15 ) {
                    
                    maskedCardNo = maskedCardNo + "***-******-";
                    maskedCardNo = maskedCardNo + cardNo.substring( 10, 14 );
                    maskedCardNo = maskedCardNo + "*";
                }
                else {
                    maskedCardNo = maskedCardNo + this.maskingText(cardNo.length -3, "*");
                    maskedCardNo = maskedCardNo + cardNo.substring( cardNo.length - 2, cardNo.length );
                }

                return maskedCardNo;
            },
            
            /**
             * 전화번호를 마스킹 처리한다.(tip:전화번호 마지막 자리는 무조건 숫자 4개로 구성)
             * 전화번호가 13자리인 경우 : 02  15776000 → 02-1577-60**
             * 전화번호가 13자리가 아닌경우 : 0215776000 → 02157760**
             *
             * @param phoneNo 마스킹 처리할 전화번호 예)02 15776000, 0215776000, 01115776000, 03115776000
             * @return 마스킹 처리된 전화번호 예)02-1577-60**, 02-1577-60**, 011-1577-60**, 031-1577-60**
             *         전화번호가 null이면 empty string,
             *         8자리 미만이거나 14보다 큰 경우
             *         지역번호 적어도 2자리, 국번 적어도 2자리, 그리고, 마지막 4자리해서 8자리
             *         02  -1577-6000가 최대 입력 길이이므로, 이보다 클 경우
             *         '-'가 없고 길이가 12자리 초과인 경우에는 입력된 전화번호를 #로 변환한 값
             */
            getPhoneNo:function(phoneNo){
                
                if ( phoneNo == null ) return "";

                phoneNo = $.trim(phoneNo);
                var lengthOfPhoneNo = phoneNo.length;

                
                if ( lengthOfPhoneNo < 8 || lengthOfPhoneNo > 14 ) {
                    return this.maskingText(lengthOfPhoneNo, "#");
                }
                    
                var maskedPhoneNo = "";
                
                if ( phoneNo.indexOf( '-' ) < 0 ) { //'-'를 포함하지 않은 경우
                    
                    //'-'를 포함하지 않은 경우 전화번호는 12자리보다 클 수 없다.
                    if ( lengthOfPhoneNo > 12 ) {
                        return this.maskingText(lengthOfPhoneNo, "#");
                    }

                    //서울 지역번호만 2자리이고 두번째자리가 2이다.
                    if ( phoneNo.indexOf("02") == 0 ) {
                        //지역번호가 세자리 이상일 경우 전화번호는 적어도 8자리 이상이여야 한다.
                        if ( lengthOfPhoneNo < 8 ) {
                            return this.maskingText(lengthOfPhoneNo, "#");
                        }

                        maskedPhoneNo += "02-";
                        maskedPhoneNo += $.trim(phoneNo.substring( 2, lengthOfPhoneNo - 4 ));
                        maskedPhoneNo += "-";
                        maskedPhoneNo += $.trim(phoneNo.substring( lengthOfPhoneNo - 4, lengthOfPhoneNo - 2 ));
                    }
                    else {
                        //지역번호가 세자리 이상일 경우 전화번호는 적어도 9자리 이상이여야 한다.
                        if ( lengthOfPhoneNo < 9 ) {
                            return this.maskingText(lengthOfPhoneNo, "#");
                        }

                        maskedPhoneNo += $.trim(phoneNo.substring( 0, 3 ));
                        maskedPhoneNo += "-";
                        maskedPhoneNo += $.trim(phoneNo.substring( 3, lengthOfPhoneNo - 4 ));
                        maskedPhoneNo += "-";
                        maskedPhoneNo += $.trim(phoneNo.substring( lengthOfPhoneNo - 4, lengthOfPhoneNo - 2 ));
                    }
                    
                } else { //'-'를 포함한 경우
                    //포함된 '-'가 2개가 아니면 즉,
                    //'-'를 delimiter로 구분한 결과로 생기는 String[]의 길이가 3이 아니면
                    //'-'를 2개 포함한 것이 아니므로 exception 발생
                    if ( (phoneNo.split( "-" )).length != 3 ) {
                        return this.maskingText(lengthOfPhoneNo, "#");
                    }

                    //전화번호 마지막 자리수가 4가 아니면 (02-1577-600 등.)
                    if ( phoneNo.lastIndexOf( '-' ) + 4 >= lengthOfPhoneNo ){
                        return this.maskingText(lengthOfPhoneNo, "#");
                    }

                    //두번째 '-'와 앞두자리까지 포함해서 substring
                    maskedPhoneNo += phoneNo.substring( 0, phoneNo.lastIndexOf( '-' ) + 3 );
                }

                maskedPhoneNo += "**";

                return maskedPhoneNo;
                
            },
            /**
             * 성명 문자열을 마스킹 처리한다.
             * (성명 중 명에 해당하는 이름의 첫 글자를 마스킹 처리)
             *
             * @param originalString 마스킹할 성명 문자열
             * @return
             */
            maskName:function(originalString){
                var maskedString = originalString;
                
                if (maskedString.length == 2) { // 이름이 한 글자 인 경우
                    maskedString = maskedString.substring(0, 1) + this.maskingText(1, "*");
                }
                else if (maskedString.length == 3) { // 이름이 두 글자 인 경우
                    maskedString = maskedString.substring(0, 1) + this.maskingText(1, "*") + maskedString.substring(2);
                }
                else if (maskedString.length > 3) { // 이름이 세 글자 이상이 경우
                    maskedString = maskedString.substring(0, 1) + this.maskingText(2, "*") + maskedString.substring(3, maskedString.length);
                }

                return maskedString;
            },
            
            /**
             * 이메일을 마스킹 처리
             *
             * @param originalString 마스킹할 이메일주소 문자열
             * @return
             */
            maskEmail:function(originalString){

                if (originalString.indexOf("@") < 0) {
                    return originalString;
                }

                var maskedString = originalString;

                var mMaskedStrings = maskedString.split("@");
                
                if (
                        mMaskedStrings[0] == null || mMaskedStrings[0] == "" || mMaskedStrings[0] == undefined || mMaskedStrings[0] == "undefined" ||
                        mMaskedStrings[1] == null || mMaskedStrings[1] == "" || mMaskedStrings[1] == undefined || mMaskedStrings[1] == "undefined"
                   ) {
                    return maskedString;
                }

                if (mMaskedStrings[0].length > 3) { // 최소 마스킹 길이보다 길 경우
                    maskedString = mMaskedStrings[0].substring(0, mMaskedStrings[0].length - 3) + this.maskingText(3, "*") + "@"
                            + mMaskedStrings[1];
                } else {
                    maskedString = this.maskingText(3, "*") + "@" + mMaskedStrings[1];
                }

                return maskedString;
            },
            /**
             * 아이디을 마스킹 처리
             *
             * @param maskedString 마스킹할 아이디 문자열
             * @return
             */
            maskID:function(originalString){
                var maskedString = originalString;

                if (maskedString.length > 3) { // 최소 마스킹 길이보다 길 경우
                    maskedString = maskedString.substring(0, maskedString.length - 3) + this.maskingText(3, "*");
                }
                else {
                    maskedString = this.maskingText(maskedString.length, "*");
                }

                return maskedString;
            },
            /**
             * @param LENGTH
             * @param charText
             * @return LENGTH값 만큼 charText 문자열을 리턴한다.
             */
            maskingText:function(lengthOfPhoneNo, charText){
                
                if(charText == null || charText == undefined || charText == "undefined" || charText == "") {
                    charText = "*";
                }
                
                var maskingTextTemp = "";
                
                for(i = 0 ; i < lengthOfPhoneNo ; i++) {
                    maskingTextTemp += charText;
                }
                
                return maskingTextTemp;
            },
            /**
             * 쿠키 값을 가져온다.
             *
             * @param cName 쿠키 이름
             * @return 쿠키값
             */
            getCookie:function(cName) {
                cName = cName + '=';
                var cookieData = document.cookie;
                var start = cookieData.indexOf(cName);
                var cValue = '';
                if(start != -1){
                     start += cName.length;
                     var end = cookieData.indexOf(';', start);
                     if(end == -1)end = cookieData.length;
                     cValue = cookieData.substring(start, end);
                }
                return unescape(cValue);
           }          
        }
    });
})(jQuery);