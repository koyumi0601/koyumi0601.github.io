(function($, window, document, undefined) {

    /**
     * 
     * $("form").hcValid({ options }) options callback: 콜백 함수
     * 
     * submitBtn : validation 을 동작시킬 버튼, anchor, image 등등
     * 
     * propertiesPath: validation 메시지 파일 경로
     * 
     * propertiesFileName: validation 파일 이름
     * 
     * 
     * 
     * 1. 기본 처리
     * 
     * $("#certForm").hcValidate({
     * 
     * submitBtn:"confirmBtn",
     * 
     * callback:function(obj, ev){
     * 
     * $("#certForm").attr("action", "xxx.hc");
     * 
     * $("#certForm").submit(); }
     * 
     * });
     * 
     * 
     * 
     * 
     * 
     * 2. 동적 처리
     * 
     * var valid = function() {
     * 
     * $("#certForm").hcValidate({
     * 
     * submitBtn:"confirmBtn",
     * 
     * callback:function(obj, ev){
     * 
     * $("#certForm").attr("action", "xxx.hc");
     * 
     * $("#certForm").submit(); }
     * 
     * }); }
     * 
     * //load
     * 
     * valid();
     * 
     * 
     * 
     * //condition
     * 
     * if(isView){ //동적으로 화면이 변경되어 다시 유효성 체크를 해야되는 경우
     * 
     * valid(); }
     * 
     * <form name="certForm" id="certForm">
     * 
     * ....
     * 
     * <input type="button" name="confirmBtn" id="confirmBtn"/>
     * 
     * </form>
     * 
     * 
     * 
     */

    var JqueryValidate = function(elem, options) {
        this.elem = null;
        this.$elem = null;
        this.options = null;
        this.metadata = null;
        this.$validInputs = null;
        this.elem = elem;
        this.$elem = $(elem);
        this.options = options;
        this.metadata = this.$elem.data('plugin-options');
        this.$validInputs = this.$elem.find('input[valid],select[valid],textarea[valid]');
        this.notError = true;
    };

    JqueryValidate.prototype = {
        rules : {
            // -----> rules : validation 룰
            required : 'required', // "requird":"true"
            msgGb : '', // "msgGb":"pop"(팝업 노출), "both"(text and 팝업 노출), 값이 없으면 text 노출
            requiredCheck : 'required_check',
            email : 'email',
            email2 : 'email2',
            digits : 'digits',
            max : 'max',
            min : 'min',
            minValue : 'minValue', // 이우형 _GGBM07 새로추가
            excess : 'excess', // 이우형 _GGBM07 새로추가
            equal : 'equal',
            numEqual : 'numEqual',  //특수문자를 제외하고 숫자만 비교           
            between : 'between',
            password : 'password',
            custom : 'custom',
            confirm : "confirm",
            dateTerm : "dateTerm",
            hidden : "hidden",
            onAfterBlur : "onAfterBlur",
            onAfterClick : "onAfterClick",
            onAfterKeyup : "onAfterKeyup",
            customFunc : "customFunc"
        },

        fixed : {
            // fix 된 값들 - 수정하지 않는다.
            msgValidateClass : 'validate',
            validAttr : 'valid',
            errorClass : 'error',
            errorElement : 'p',
            initInputMessageClass : '_Message', // 초기 validate 항목의 error 메세지 표시할
            // div 클래스 명
            initInputMsg : [], // setInitInputMessage 호출시 사용 : validate 항목의
            // error 메세지 데이터
            initSenseReaderMsg : [],
            firstErrorMsg : '', // submit 시 첫번째 에러 메세지 alert 표시용
            msgProp : '',
            onAfterkeyBlurReturn : '',
            onAfterClickReturn : '',
            onAfterKeyupReturn : '',
            customFuncReturn : false,
            isErrorReturn : false
        },

        defaults : {
            callback : '',
            // -------> message resource
            propertiesPath : '/docfiles/resources/pc/js/properties/',
            propertiesFileName : 'validation-messages',
            // -------> Error Process
            initMsgFlg : true, // setInitInputMessage 호출시만 사용
            // --------> GROUP
            submitBtn : null
        },

        // 최대 기간을 넣어 그 이상의 범위로 검색할 수 없도록 제한한다.
        // valid:{"dateTerm": "startDate, endDate, 7D"}
        dateRange : {
            _7D : { term : "7", msg : "1주일" },
            _1M : { term : "1", msg : "1개월" },
            _2M : { term : "2", msg : "2개월" },
            _3M : { term : "3", msg : "3개월" },
            _6M : { term : "6", msg : "6개월" },
            _1Y : { term : "12", msg : "1년" },
            _2Y : { term : "24", msg : "2년" }
        },

        /**
         * 
         * hcValid 함수가 처음으로 호출하는 함수
         * 
         * 사용자 변수를 내부에 전달.
         * 
         * 메시지 객체 생성.
         * 
         * submit 버튼 또는 submit에 준하는 button/a 객체가 있을 경우 해당 객체 클릭시 validate 실행
         * 
         * 각 입력필드에 blur 이벤트 시 validate 실행
         * 
         * 페이지 최초 로딩시에 센스리더용 텍스트 생성
         * 
         */

        init : function() {
            var self = this;
            var propertiesName;
            self.config = $.extend({}, self.defaults, self.options, self.metadata);
            propertiesName = self.config.propertiesFileName;
            // 메시지 객체 생성
            $.hc.messages.setProperties({
                name : propertiesName,
                path : self.config.propertiesPath,
                mode : "map"
            });
            self.fixed.msgProp = $.hc.messages;
            /**
             * 
             * Form인 경우 if(self.$elem[0].tagName == "FORM") {
             * 
             * self.$elem.off('submit.jqueryValidate');
             * 
             * self.$elem.on('submit.jqueryValidate', $.proxy(self.handleSubmit,
             * 
             * self)); }
             * 
             */

            // Group인 경우
            // 동적인 호출이 있을 경우를 대비해 click 이벤트를 off 시킨다.
            // submitBtn은 일반적으로 반드시 있어야 한다.
            if (self.config.submitBtn != null) {
                if (!_.isArray(self.config.submitBtn)) {
                    if (_.isString(self.config.submitBtn)) {
                        if ($("#" + self.config.submitBtn).length > 0) {
                            if ("SELECT" == $("#" + self.config.submitBtn).get(0).tagName) {
                                $("#" + self.config.submitBtn).off('change.jqueryValidate');
                                $("#" + self.config.submitBtn).on('change.jqueryValidate', $.proxy(self.handleSubmit, self));
                            } else {
                                $("#" + self.config.submitBtn).off('click.jqueryValidate');
                                $("#" + self.config.submitBtn).on('click.jqueryValidate', $.proxy(self.handleSubmit, self));
                            }
                        } else {
                            // alert("Error : submit 버튼이 있는지 확인해주세요.");
                        }
                    } else {
                        if ("SELECT" == self.config.submitBtn.get(0).tagName) {
                            self.config.submitBtn.off('change.jqueryValidate');
                            self.config.submitBtn.on('change.jqueryValidate', $.proxy(self.handleSubmit, self));
                        } else {
                            self.config.submitBtn.off('click.jqueryValidate');
                            self.config.submitBtn.on('click.jqueryValidate', $.proxy(self.handleSubmit, self));
                        }
                    }
                } else {
                    var els = self.config.submitBtn;
                    for (var idx = 0; idx < els.length; idx++) {
                        var el = els[idx];
                        if (_.isString(el)) {
                            if ("SELECT" == $("#" + el).get(0).tagName) {
                                $("#" + el).off('change.jqueryValidate');
                                $("#" + el).on('change.jqueryValidate', $.proxy(self.handleSubmit, self));
                            } else {
                                $("#" + el).off('click.jqueryValidate');
                                $("#" + el).on('click.jqueryValidate', $.proxy(self.handleSubmit, self));
                            }
                        } else {
                            if ("SELECT" == el.get(0).tagName) {
                                el.off('change.jqueryValidate');
                                el.on('change.jqueryValidate', $.proxy(self.handleSubmit, self));
                            } else {
                                el.off('click.jqueryValidate');
                                el.on('click.jqueryValidate', $.proxy(self.handleSubmit, self));
                            }
                        }
                    }
                }
            } else {
                // submitBtn이 없는경우 예외처리
                // 만약 submit 이라는 키워드를 가진 element가 있다면 submit 버튼으로 대체 한다.
                if (self.$elem.find(":submit").length > 0) {
                    self.$elem.find(":submit").off("click.jqueryValidate");
                    self.$elem.find(":submit").on('click.jqueryValidate', $.proxy(self.handleSubmit, self));
                } else {
                    // 그래도 버튼이 없으면 에러 메시지..출력
                    // alert($.hc.messages.get("alert.valid.submitBtn.notfound"));
                    return false;
                }
            }
            // }
            // 폼 input 필드 별 체크
            self.$validInputs.off('keyup.jqueryValidate');
            self.$validInputs.on('keyup.jqueryValidate', function(event) {
                self.handleInputFocus(event, $(event.target));
            });
            self.$validInputs.off('change.jqueryValidate');
            self.$validInputs.filter("select").on('change.jqueryValidate', function(event) {
                // 2016.04.25 모든 인풋에서 에러 발생으로 인해 주석처리
                // 2016.06.06 select 태그만 re-활성화
                self.handleInputFocus(event, $(this));
            });
            $.each(self.$validInputs, function(k, el) {
                if ($(el).attr("type") == "radio" || $(el).attr("type") == "checkbox" || $(el).attr("type") == "date") {
                    $("input[name=" + $(el).attr("name") + "]").on('change.jqueryValidate', function(event) {
                        // console.log($(event.target));
                        //self.handleInputFocus(event, $(el));
                        self.removeErrorMsg($(el));
                        self.allErrorCheck();
                    });
                }
            });
            // 화면 로드시 폼체크 항목 list 표시
            if (self.config.initMsgFlg) {
                self.setInitInputMessage(self.$validInputs);
            }
            self.setInitInputEvent();
            //console.log('%c■■■■ JqueryValidate','color: teal', self);
            return this;
        },

        /**
         * 
         * submit 버튼 클릭시 이벤트 handle
         * 
         * 유효성 체크, 만족하면 callback 함수 실행.
         * 
         * 만족하지 않으면 callback 함수 실행하지 않고 에러 출력
         * 
         * @param e
         * 
         */

        handleSubmit : function(e) {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            this.hasError = false;
            // console.log(this.$validInputs);
            self.$elem.find(":submit").off("submit");
            e.preventDefault();
            e.stopPropagation();
            self.removeAllErrorMsg(); // 에러메세지 삭제
            // 에러 존재시 submit 안함
            if (!self.validateFields()) {
                if (typeof config.callback == "function" && config.callback != '') { // callback
                    config.callback.call(self, self.$elem, e, e.currentTarget);
                }
            }
            return;
        },

        /**
         * 
         * 입력 필드 keyup시 이벤트 handle
         * 
         * 유효성 체크, 만족하면 에러메시지 삭제
         * 
         * 만족하지 않으면 해당 에러메시지 출력
         * 
         * @param element
         * 
         */

        handleInputFocus : function(ev, element) {
            var self = this;
            var requiredFlg = false;
            var fixed = self.fixed;
            var config = self.config;
            var hasError = false;
            var ret = false;
            var message = "";
            config.initMsgFlg = false;
            var jsonValid = self.jsonReplace(element.attr(fixed.validAttr));
            var validObj = $.parseJSON(jsonValid);
            self.removeErrorMsg(element); // 에러메세지 삭제

            $.each(validObj, function(rule, option) {
                if (self._validate(element, rule, option)) {
                    hasError = true;
                }
            });

            $.each(validObj, function(key, option) {
                var hasError = false;
                switch (key) {
                    // 커스터마이징 된 에러를 출력해야 하므로 에러메시지 키를 받아야 한다.
                    // return이 true 이면 removeError
                    // return이 false 이면 addError
                    case self.rules.onAfterBlur:
                        if (!hasError && ev.type == 'keyup') {
                            if (option.indexOf(".") >= 0) {
                                fn = eval("window." + option);
                                config.onAfterkeyBlurReturn = fn(ev, element);
                            } else {
                                config.onAfterkeyBlurReturn = window[option](ev, element); // 문자
                            }

                            if (typeof config.onAfterkeyBlurReturn != "undefined") {
                                if (typeof config.onAfterkeyBlurReturn == "object" && config.onAfterkeyBlurReturn.result == "error") {

                                    if (!config.onAfterkeyBlurReturn.message) {
                                        message = self.fixed.msgProp.get(config.onAfterkeyBlurReturn.key);
                                    } else {
                                        message = config.onAfterkeyBlurReturn.message;
                                    }
                                    self.addErrorText(element, message);
                                    hasError = true;
                                } else {
                                    self.removeErrorMsg(element);
                                    hasError = true;
                                }
                            }
                        }
                        break;
                    case self.rules.customFunc:
                        if (!hasError && ev.type == 'keyup') {
                            if (option.indexOf(".") >= 0) {
                                fn = eval("window." + option);
                                config.customFuncReturn = fn(ev, element);
                            } else {
                                config.customFuncReturn = window[option](ev, element); // 문자
                            }

                            if (typeof config.customFuncReturn != "undefined") {
                                message = self.fixed.msgProp.get("valid.msg.customFunc." + config.customFuncReturn);
                                self.addErrorText(element, message);
                            } else {
                                self.removeErrorMsg(element);
                            }
                        }
                        break;
                    default:
                        break;
                }
            });
            self.allErrorCheck(element, hasError);
        },

        allErrorCheck : function(element, hasError) {
            var self = this;
            var fixed = self.fixed;
            var allCheck = false;
            if (hasError) {
                allCheck = hasError;
            } else {
                self.$validInputs.not(element).each(function() {
                    var _this = this;
                    if ($(_this).attr(fixed.validAttr)) {
                        self.notError = false;
                        $.each($.parseJSON(self.jsonReplace(String($(_this).attr(fixed.validAttr)))), function(rule, option) {
                            if (!allCheck) {
                                if (self._validate($(_this), rule, option)) {
                                    allCheck = true;
                                } else {
                                    allCheck = false;
                                }
                            }
                        });
                        self.notError = true;
                    }
                });
            }

            var submitBtn = $(document.getElementById(self.config.submitBtn));
            if (submitBtn.size() && allCheck) {
                if (!submitBtn.hasClass('disabled')) {
                    submitBtn.addClass('disabled');
                }
            } else {
                if (submitBtn.hasClass('disabled')) {
                    submitBtn.removeClass('disabled');
                }
            }
        },
        /**
         * 
         * 에러메시지를 지정된 properties를 통해 호출한다.
         * 
         * @param checkClassName
         * 
         * @param elementLabelText
         * 
         * @param elementId
         * 
         * @param option
         * 
         * @returns {String}
         * 
         */

        msgProperties : function(checkClassName, elementLabelText, elementId, option) {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var message = '';
            switch (checkClassName) {
                case self.rules.required: // required
                    message = self.fixed.msgProp.get("valid.msg.required", elementLabelText);
                    break;
                case self.rules.email: // email
                    message = self.fixed.msgProp.get("valid.msg.email.formait");
                    break;
                case self.rules.email2: // email
                    message = self.fixed.msgProp.get("valid.msg.email.formait");
                    break;
                case self.rules.digits: // digits
                    message = self.fixed.msgProp.get("valid.msg." + option, elementLabelText);
                    break;
                case self.rules.max: // max
                    message = self.fixed.msgProp.get("valid.msg.max", elementLabelText, option);
                    break;
                case self.rules.min: // min
                    message = self.fixed.msgProp.get("valid.msg.min", elementLabelText, option);
                    break;
                case self.rules.excess: // excess 이우형
                    message = self.fixed.msgProp.get("valid.msg.excess", elementLabelText, option);
                    break;
                case self.rules.minValue: // minValue 이우형
                    message = self.fixed.msgProp.get("valid.msg.minValue", elementLabelText, option);
                    break;
                case self.rules.equal: // equal
                    message = self.fixed.msgProp.get("valid.msg.equal", elementLabelText, option);
                    break;
                case self.rules.numEqual: // numEqual
                    message = self.fixed.msgProp.get("valid.msg.numEqual", elementLabelText, option);
                    break;
                case self.rules.password: // password
                    message = self.fixed.msgProp.get("valid.msg.password." + option, elementLabelText);
                    break;
                case self.rules.custom: // Custom
                    if (option == 'custom1') {
                        message = self.fixed.msgProp.get("valid.msg.custom.noSpecial");
                    } else if (option == "idVerify") {
                        message = self.fixed.msgProp.get("valid.msg.custom." + option, elementLabelText);
                    }
                    break;
                case self.rules.confirm:
                    message = self.fixed.msgProp.get("valid.msg.confirm");
                    break;
                case self.rules.between:
                    var opt = option.split("~");
                    message = self.fixed.msgProp.get("valid.msg.between", elementLabelText, $.trim(opt[0]), $.trim(opt[1]));
                    break;
                case self.rules.dateTerm:
                    message = self.fixed.msgProp.get("valid.msg.dateTerm." + option.key, elementLabelText, option.vars, option.vars2);
                    break;
                case self.rules.hidden:
                    message = self.fixed.msgProp.get("valid.msg.hidden", elementLabelText);
                    break;
                case self.rules.customFunc:
                    message = self.fixed.msgProp.get("valid.msg.customFunc." + option, elementLabelText);
                    break;
                default:
                    break;
            }

            if (fixed.firstErrorMsg == '') {
                fixed.firstErrorMsg = message;
            }
            return message;
        },

        /**
         * 
         * 에러 항목 라벨 취득
         *  - title 지정시 title 값 사용
         *  - title 미지정시 같은 레벨의 label 값 사용
         *  - 모두 미지정시 id 값 사용
         * 
         * @param element
         * 
         * @returns
         * 
         */

        errorLabelText : function(element) {
            var label = element.attr('title');
            var id = element.attr('id');
            if (typeof label == 'undefined') {
                if ($("label[for='" + id + "']").length > 0) {
                    label = $("span", $("label[for='" + id + "']")[0].outerHTML).remove().end().html();
                }
            }
            // return (typeof label == 'undefined' || label == '') ?
            // element.attr('id') : label;
            return (typeof label == 'undefined' || label == '') ? '' : label;
        },

        // return (typeof label == 'undefined' || label == '') ?
        // element.attr('id') : label;

        /**
         * 
         * 초기 입력항목 체크항목 표시 (센스리더)
         * 
         * @param elements
         * 
         */

        setInitInputMessage : function(elements) {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var message = '';
            var propKeys = self.fixed.msgProp.getPropertiesKeys("valid.msg");
            var labelText = "";
            var first = true;
            var passedMessage = false;
            self.$validInputs.each(function(idx) {
                var $field = self.$validInputs.eq(idx);
                var elementId = $field.attr("id");
                var jsonValid = self.jsonReplace($field.attr(fixed.validAttr));
                var validObj = $.parseJSON(jsonValid);
                labelText = self.errorLabelText($field);
                // var validObj =
                // $.parseJSON($field.attr(fixed.validAttr));
                var msgData = {
                    value : []
                };
                var div = "";
                if ($("#" + elementId + fixed.initInputMessageClass).length <= 0) {
                    $.each($("#container").find(".vdBox"), function(i, obj) {
                        if ($(obj).find($field).length > 0) {
                            div = $("<div></div>").attr("id", elementId + fixed.initInputMessageClass).attr("class", "description");
                            $(obj).prepend(div);
                        }
                    });
                    $.each(validObj, function(key, options) {
                        var preLabel = "";
                        var option = "";
                        var labelText2 = labelText;
                        // console.log(labelText2,",",key);
                        $.each(propKeys, function(i, name) {
                            var message = "";
                            if (name.indexOf(key) >= 0) {
                                if (key == self.rules.excess) { // 수정_이우형
                                    message = self.fixed.msgProp.get(name, labelText2);// 수정_이우형
                                } // 수정_이우형
                                if (key == self.rules.minValue) { // 수정_이우형
                                    message = self.fixed.msgProp.get(name, labelText2);// 수정_이우형
                                } // 수정_이우형
                                if (key == self.rules.between) {
                                    var opt = options.split("~");
                                    message = self.fixed.msgProp.get(name, labelText2, $.trim(opt[0]), $.trim(opt[1]));
                                } else if (key == self.rules.password) {
                                    if (name.indexOf(options) > 0) {
                                        message = self.fixed.msgProp.get(name, labelText2);
                                    }
                                } else if (key == self.rules.dateTerm) {
                                    var form = $field.closest(".direct_cld"); // .closest(".vdForm");
                                    var span = form.find(".vd-title"); // .find(".vdTitle");
                                    if (span.length > 0) {
                                        labelText2 = span.text();
                                    }
                                    message = self.getDateTermMessage(name, labelText2, options);
                                } else if (key == self.rules.customFunc) {
                                    message = self.fixed.msgProp.get("valid.msg.customFunc." + options, labelText2);
                                } else if (key == self.rules.custom) {
                                    if (name.indexOf(options) > 0) {
                                        message = self.fixed.msgProp.get("valid.msg.custom." + options, labelText2);
                                    }
                                } else {
                                    message = self.fixed.msgProp.get(name, labelText2, options);
                                }
                                if (message != "") {
                                    if (preLabel != labelText2) {
                                        if ($('#' + elementId + fixed.initInputMessageClass).length > 0) {
                                            $('#' + elementId + fixed.initInputMessageClass).html("<ul><li>" + message + "</li></ul>");
                                        }
                                    } else {
                                        $('#' + elementId + fixed.initInputMessageClass).find("ul").append("<li>" + message + "</li>");
                                    }
                                }
                            }
                            preLabel = labelText2;
                        });
                    });
                }
            });
            config.initMsgFlg = false; // 표시후 다른 체크에 영향 없도록 플래그 값 변경
        },

        /**
         * 
         * 특정 필드에 대해 keyup, click 이벤트 발생시 유효성 체크 후
         * 
         * 지정된 callback 함수를 호출
         * 
         */

        setInitInputEvent : function() {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            self.$validInputs.each(function(idx) {
                var $field = self.$validInputs.eq(idx);
                var jsonValid = self.jsonReplace($field.attr(fixed.validAttr));
                var validObj = $.parseJSON(jsonValid);
                // 각 이벤트 별로 에러처리...
                // 모든 상태를 만족하는 경우에 실행될 것인가??
                $.each(validObj, function(key, option) {
                    switch (key) {
                        case self.rules.onAfterClick:
                            $field.off("click." + key);
                            $field.on("click." + key, function(e) {
                                if (option.indexOf(".") >= 0) {
                                    fn = eval("window." + option);
                                    config.onAfterClickReturn = fn(e, $field);
                                } else {
                                    config.onAfterClickReturn = window[option](e, $field); // 문자
                                }
                            });
                            break;
                        case self.rules.onAfterKeyup:
                            $field.off("keyup." + key);
                            $field.on("keyup." + key, function(e) {
                                if (option.indexOf(".") >= 0) {
                                    fn = eval("window." + option);
                                    config.onAfterKeyupReturn = fn(e, $field);
                                } else {
                                    config.onAfterKeyupReturn = window[option](e, $field); // 문자
                                }
                            });
                            break;
                        default:
                            break;
                    }
                });
            });
            config.initMsgFlg = false; // 표시후 다른 체크에 영향 없도록 플래그 값 변경
        },

        /**
         * 
         * form 의 전체 입력필드 체크
         * 
         * 
         * 
         * @param index :
         * 
         * 지정된 폼의 required지정된 input 항목의 index
         * 
         */

        validateFields : function() {
            var self = this;
            var fixed = this.fixed;
            var config = this.config;
            var firstErrorField = null;
            var hasErrorFlg = false;
            var checkedRequired = false;
            var msgPopupFlg = false;

            self.$validInputs.each(function(idx) {
                var $field = self.$validInputs.eq(idx);
                var jsonValid = self.jsonReplace($field.attr(fixed.validAttr));
                var validObj = $.parseJSON(jsonValid);
                // var validObj = $.parseJSON($field.attr(fixed.validAttr));
                // required만 먼저 체크하면 다음은 안해도 되니까..
                if (validObj.hasOwnProperty(self.rules.required) && validObj.required) {
                    if (self._validate($field, self.rules.required) && firstErrorField == null) {
                        firstErrorField = $field;
                        hasErrorFlg = true;
                        checkedRequired = true;
                        //console.log("validObj.msgGb:" + validObj.msgGb +", $field:" +$field.attr("id") );
                        if (validObj.msgGb == "pop" || validObj.msgGb == "both") {
                            msgPopupFlg = true;
                        }
                    }
                }

                // 필드에 필수입력(선택)값이 있을시에 나머지 validate 처리
                // var fieldValue = self.elementValue($field);
                if (!hasErrorFlg) {
                    $.each(validObj, function(rule, option) {
                        if (self._validate($field, rule, option, checkedRequired)) {
                            if (firstErrorField == null) {
                                firstErrorField = $field;
                                if (validObj.msgGb == "pop" || validObj.msgGb == "both") {
                                    msgPopupFlg = true;
                                }
                            }
                            hasErrorFlg = true;
                        }
                    });
                }
            });

            if (config.initMsgFlg) {
                hasErrorFlg = false; // 초기 validate 메세지 설정용 일시 체크 안함
            }
            
            //console.log("msgPopupFlg:"+msgPopupFlg+", hasErrorFlg:" + hasErrorFlg+ ", firstErrorField :"+firstErrorField);
            // error가 있으면 첫번째 에러 항목으로 포커스 이동 및 alert 표시
            if (hasErrorFlg && firstErrorField) {
                // display가 none일시 focus이동 하면 에러가 난다.
                if (firstErrorField.css("display") != "none") {
                    if (!firstErrorField.is(':visible')) {
                        // 약관 아코디언 닫혀 있으면 강제로 열기
                        var agreeWrap = firstErrorField.closest('div.agree_wrap');
                        if (agreeWrap.size()) {
                            agreeWrap.find('button.btn_aco').trigger('click');
                        }
                    } 
                    //var offsetTop = firstErrorField[0].offsetTop || 0;
                    //$(window).scrollTop(offsetTop); // IE에서 위치 조정을 위해 추가
                    // firstErrorField.focus();
                    openMsgPopupFocus();
                } else {
                    var height = firstErrorField.css("height");
                    firstErrorField.css("height", "0px");
                    firstErrorField.css("display", "inline");
                    $(window).scrollTop(0); // IE에서 위치 조정을 위해 추가
                    // firstErrorField.focus();
                    openMsgPopupFocus();
                    $(window).scrollTop($(window).scrollTop() + 30);
                    firstErrorField.css("display", "none");
                    firstErrorField.css("height", height + "px");
                }

                function openMsgPopupFocus() {
                    // console.log("msgPopupflg :::::::::: " + msgPopupFlg);
                    var label = $('label[for="' + firstErrorField[0].id + '"]');
                    if (msgPopupFlg) { // 팝업 노출인 경우
                        openAlertType1(fixed.firstErrorMsg, '', function() {
                            /checkbox/.test(firstErrorField[0].type) && label.size() ? label.focus() : firstErrorField.focus();
                        });
                    } else {
                        /checkbox/.test(firstErrorField[0].type) && label.size() ? label.focus() : firstErrorField.focus();
                    }
                }
            }

            return hasErrorFlg;

        },

        /**
         * 
         * 각 이벤트 별로 에러처리
         * 
         * required, digit, min, max.... 유효성 체크 타입별로 처리를 분리하는 내부 호출 함수
         * 
         * @param element
         * 
         * @param validateRule
         * 
         * @param option
         * 
         * @param requiredFlag
         * 
         * @returns {Boolean}
         * 
         */

        _validate : function(element, validateRule, option, requiredFlag) {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var hasError = false;
            var fieldValue = self.elementValue(element);
            var elementId = element.attr("id");
            if (element.is(':disabled')) {
                self.removeErrorMsg(element);
                return hasError;
            }
            switch (validateRule) {
                case self.rules.required: // required
                    if (typeof requiredFlag == "undefined" || !requiredFlag) {
                        if (element.attr("type") == self.rules.hidden) {
                            hasError = self.validateHiddenRequiredField(element);
                        } else {
                            hasError = self.validateRequiredField(element);
                        }
                    }
                    break;
                case self.rules.email: // email
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateEmail(element);
                    break;
                case self.rules.email2: // email
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateEmail2(element);
                    break;
                case self.rules.digits: // digits
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateDigits(element, option);
                    break;
                case self.rules.max: // max
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateMaxSize(element, option);
                    break;

                case self.rules.min: // mix
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateMinSize(element, option);
                    break;

                case self.rules.excess: // excess 이우형 _ggbm07
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateExcess(element, option);
                    break;

                case self.rules.minValue: // minValue 이우형 _ggbm07
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateMinValue(element, option);
                    break;
                case self.rules.equal: // equal
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateEqual(element, option);
                    break;
                case self.rules.numEqual: // numEqual
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateNumEqual(element, option);
                    break;
                case self.rules.password: // password
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue)) {
                        hasError = self.validatePassword(element, option);
                    }
                    break;
                case self.rules.custom: // Custom
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateCustom(element, option);
                    break;
                case self.rules.confirm: // Confirm
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateConfirm(element, option);
                    break;
                case self.rules.between: // Between
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue))
                        hasError = self.validateBetween(element, option);
                    break;
                case self.rules.dateTerm: // dateTerm
                    if (config.initMsgFlg || self.isValueNotNull(fieldValue)) {
                        hasError = self.validateDateTerm(element, option);
                    }
                    break;
                case self.rules.customFunc:
                    hasError = self.validateCustomFunc(element, option);
                    break;
                default:
                    break;
            }
            return hasError;
        },

        /**
         * 
         * 특정 값을 체크(라디오, 체크박스)했는지 알기 위해
         * 
         * hidden 필드에 값을 셋팅한 경우 유효성 체크
         * 
         * @param element
         * 
         * @returns {Boolean}
         * 
         */

        validateHiddenRequiredField : function(element) {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var hasError = false;
            var jsonValid = self.jsonReplace(element.attr(fixed.validAttr));
            var validObj = $.parseJSON(jsonValid);
            var elementId = element.attr("id");
            var targetEl = [];
            if (validObj.required.indexOf(",") > 0) {
                targetEl = validObj.required.split(",");
            } else {
                targetEl[0] = validObj.required;
            }
            if (validObj.hasOwnProperty(self.rules.required) && targetEl.length > 0) {
                var fieldValue = self.elementValue(element);
                var labelText = self.errorLabelText(element);
                var errorMsg = '';
                if (typeof fieldValue == "undefined" || fieldValue === '') {
                    errorMsg = self.msgProperties(self.rules.hidden, labelText, elementId);
                    $.each(targetEl, function(i, id) {
                        $("#" + $.trim(id)).off("click.valid");
                        $("#" + $.trim(id)).on("click.valid", {
                            "self" : element
                        }, function(e) {
                            e.data.self.val("checked");
                            self.removeErrorMsg(e.data.self);
                        });
                    });
                    hasError = true;
                }
                // 에러메세지 표시
                self.addErrorText(element, errorMsg, $("#" + $.trim(targetEl[0])));
            } else {
                hasError = false;
            }
            return hasError;
        },

        /**
         * 
         * required 체크
         * 
         * 
         * 
         * @param element
         * 
         * @return 필수체크 에러시 true 반환
         * 
         */

        validateRequiredField : function(element, addErrorNot) {
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var hasError = false;
            var jsonValid = self.jsonReplace(element.attr(fixed.validAttr));
            var validObj = $.parseJSON(jsonValid);
            var elementId = element.attr("id");
            if (validObj.hasOwnProperty(self.rules.required) && validObj.required) {
                var fieldValue = self.elementValue(element);
                var labelText = self.errorLabelText(element);
                var errorMsg = '';
                if (fieldValue === '') {
                    errorMsg = self.msgProperties(self.rules.required, labelText, elementId);
                    hasError = true;
                }
                if (element[0].nodeName.toLowerCase() === "select") {
                    var val = $(element).val();
                    if (val && val.length <= 0) {
                        errorMsg = self.msgProperties(self.rules.required, labelText, elementId);
                        hasError = true;
                    }
                }
                if ((/radio|checkbox|select/i).test(element[0].type)) {
                    self.removeErrorMsg(element);
                    if (self.getLength(element, fieldValue) <= 0) {
                        errorMsg = self.msgProperties(self.rules.required, labelText, elementId);
                        hasError = true;
                    }
                }
                if (/select/i.test(element[0].type)) {

                }
                // 에러메세지 표시
                self.addErrorText(element, errorMsg);
            } else {
                hasError = false;
            }
            return hasError;
        },

        /**
         * 
         * email 유효성 체크
         * 
         * 
         * 
         * @param element
         * 
         * @returns {Boolean}
         * 
         */

        validateEmail : function(element) {
            var self = this;
            var fieldValue = self.elementValue(element);
            var elementId = element.attr("id");
            // if
            // (!(/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$/.test(fieldValue.toLowerCase())))
            // {
            if (!(/^([a-z0-9-]+)(\.[a-z0-9-]+)*(\.[a-z]{2,4})$/.test(fieldValue.toLowerCase()))) {
                var labelText = self.errorLabelText($(element));
                errorMsg = self.msgProperties(self.rules.email, labelText, elementId);
                // 에러메세지 표시
                self.addErrorText(element, errorMsg);
                return true;
            } else {
                return false;
            }
        },

        /**
         * 
         * email 유효성 체크2
         * 
         * 
         * 
         * @param element
         * 
         * @returns {Boolean}
         * 
         */

        validateEmail2 : function(element) {
            var self = this;
            var fieldValue = self.elementValue(element);
            var elementId = element.attr("id");
            if (!(/^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/.test(fieldValue))) {
                // if
                // (!(/^([a-z0-9-]+)(\.[a-z0-9-]+)*(\.[a-z]{2,4})$/.test(fieldValue.toLowerCase())))
                // {
                var labelText = self.errorLabelText($(element));
                errorMsg = self.msgProperties(self.rules.email, labelText, elementId);
                // 에러메세지 표시
                self.addErrorText(element, errorMsg);
                return true;
            } else {
                return false;
            }
        },

        /**
         * 
         * 숫자 이외의 문자 체크
         * 
         * 
         * 
         * @param element
         * 
         * @returns {Boolean}
         * 
         */

        validateDigits : function(element, option) {
            var self = this;
            var fieldValue = self.elementValue(element);
            var elementId = element.attr("id");
            var flagType = "";
            if (!(/(^\d{1,3}(\.?\d{3})*(,\d{2})?$)|(^\d{1,3}(,?\d{3})*(\.\d{2})?$)/.test(fieldValue))) {
                flagType = (option == "true") ? "digits" : option;
            }
            if (flagType != "") {
                var labelText = self.errorLabelText($(element));
                errorMsg = self.msgProperties(self.rules.digits, labelText, elementId, flagType);
                // 에러메세지 표시
                self.addErrorText(element, errorMsg);
                return true;
            } else {
                return false;
            }
        },

        /**
         * 
         * 문자열 최대 입력 제한 체크
         * 
         * 
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateMaxSize : function(element, option) {
            var self = this;
            var max = option;
            var fieldValue = self.elementValue(element);
            var len = self.getLength(element, fieldValue);
            var elementId = element.attr("id");
            if (len > max) {
                var labelText = self.errorLabelText($(element));
                errorMsg = self.msgProperties(self.rules.max, labelText, elementId, option);
                // 에러메세지 표시
                self.addErrorText(element, errorMsg);
                return true;
            } else {
                return false;
            }
        },

        /**
         * 
         * 문자열 최소 입력 제한 체크
         * 
         * 
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateMinSize : function(element, option) {
            var self = this;
            var min = option;
            var len = self.getLength(element, self.elementValue(element));
            var elementId = element.attr("id");
            if (len < min) {
                var labelText = self.errorLabelText($(element));
                errorMsg = self.msgProperties(self.rules.min, labelText, elementId, option);
                // 에러메세지 표시
                self.addErrorText(element, errorMsg);
                return true;
            } else {
                return false;
            }
        },

        /**
         * 
         * 최소 사용가능한 X캐시백 _ 이우형 GGBM07
         * 
         * 
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateExcess : function(element, option) {

            var self = this;

            var excess = self.elementValue(element);

            // var fieldValue = self.elementValue(element);

            var len = 10000;

            var maxCash = "10000 X캐시백";

            var elementId = element.attr("id");

            if (len > excess) {

                var labelText = self.errorLabelText($(element));

                errorMsg = self.msgProperties(self.rules.excess, labelText, elementId, maxCash);

                // 에러메세지 표시

                self.addErrorText(element, errorMsg);

                return true;

            } else {

                return false;

            }

        },

        /**
         * 
         * 현 보유 X캐시백 초과 확인 _ 이우형 GGBM07
         * 
         * 
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateMinValue : function(element, option) {

            var self = this;

            var minLen = (self.elementValue(element));

            var numMinLen = minLen - 0;

            var elementId = element.attr("id");

            var haveValue = "현 보유 X캐시백";

            var len = ($('#possPnt').val());

            var possPnt = len - 0;

            if (possPnt < numMinLen) {

                var labelText = self.errorLabelText($(element));

                errorMsg = self.msgProperties(self.rules.minValue, labelText, elementId, haveValue);

                // 에러메세지 표시

                self.addErrorText(element, errorMsg);

                return true;

            } else {

                return false;

            }

        },

        /**
         * 
         * 지정한 값과 필드의 값이 같은지 체크
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateEqual : function(element, option) {

            var self = this;

            var equal = option;

            var len = self.getLength(element, self.elementValue(element));

            var val = self.elementValue(element);

            var elementId = element.attr("id");

            var flag = false;

            if (_.isString(option) && $.isNumeric(option)) {

                if (len != equal) {

                    var labelText = self.errorLabelText($(element));

                    errorMsg = self.msgProperties(self.rules.equal, labelText, elementId, option);

                    // 에러메세지 표시

                    self.addErrorText(element, errorMsg);

                    flag = true;

                }

            } else {

                if (val != equal) {

                    var labelText = self.errorLabelText($(element));

                    errorMsg = self.msgProperties(self.rules.equal, labelText, elementId, option);

                    // 에러메세지 표시

                    self.addErrorText(element, errorMsg);

                    flag = true;

                }

            }

            return flag;

        },

        /**
         * 
         * 지정한 값과 필드의 값이 같은지 체크(특수문자 제외하고 숫자값만 비교)
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateNumEqual : function(element, option) {

            var self = this;

            var equal = option;

            var len = self.getLength(element, self.getOnlyNumber(self.elementValue(element)));

            var val = self.getOnlyNumber(self.elementValue(element));

            var elementId = element.attr("id");

            var flag = false;

            if (_.isString(option) && $.isNumeric(option)) {

                if (len != equal) {

                    var labelText = self.errorLabelText($(element));

                    errorMsg = self.msgProperties(self.rules.numEqual, labelText, elementId, option);

                    // 에러메세지 표시

                    self.addErrorText(element, errorMsg);

                    flag = true;

                }

            } else {

                if (val != equal) {

                    var labelText = self.errorLabelText($(element));

                    errorMsg = self.msgProperties(self.rules.numEqual, labelText, elementId, option);

                    // 에러메세지 표시

                    self.addErrorText(element, errorMsg);

                    flag = true;

                }

            }

            return flag;

        },

        /**
         * 
         * custom validation이 추가 될 경우 validateCustomBiz를 사용하여 처리한다.
         * 
         * @param element
         * 
         * @param option :
         *            validation option
         * 
         * @param condition :
         *            check verify condition
         * 
         * @returns {Boolean}
         * 
         * 
         * 
         */

        validateCustomBiz : function(element, option, condition) {

            var self = this;

            var flag = false;

            var elementValue = self.elementValue(element);

            var elementId = element.attr("id");

            var defaults = {

                lower : 0,

                upper : 0,

                start_upper : 0,

                alpha : 0, /* lower + upper */

                numeric : 0, // 1: 숫자 필수

                special : 0, // 1: _ 필수

                length : [ 0, Infinity ],

                custom : [ /* regexes and/or functions */],

                badWords : []

            };

            // "a123G#d"

            var re = {

                lower : /[a-z]/g, // 2 alpha : 1,

                upper : /[A-Z]/g, // 1 numeric : 1,

                start_eng : /^[A-Z]/gi, // x start_eng: 1,

                alpha : /[A-Z]/gi, // 3 special : 0,

                numeric : /[0-9]/g, // 3

                special : /[\W_]/g
            // x

            }, rule, i;

            var o = $.extend({}, defaults, condition);

            // enforce min/max length

            if (elementValue.length < o.length[0] || elementValue.length > o.length[1]) {

                flag = true;

            }

            // enforce lower/upper/alpha/numeric/special rules

            for (rule in re) {

                if (rule == "special") {

                    if (o[rule] == 0) {

                        if ((elementValue.match(re[rule]) || []).length > 0) {

                            flag = true;

                        }

                    } else {

                        if ((elementValue.match(re[rule]) || []).length <= 0) {

                            flag = true;

                        }

                    }

                } else if ((elementValue.match(re[rule]) || []).length < o[rule]) {

                    flag = true;

                }

            }

            // enforce word ban (case insensitive)

            for (i = 0; i < o.badWords.length; i++) {

                if (elementValue.toLowerCase().indexOf(o.badWords[i].toLowerCase()) > -1)

                    flag = true;

            }

            // enforce custom regex/function rules

            for (i = 0; i < o.custom.length; i++) {

                rule = o.custom[i];

                if (rule instanceof RegExp) {

                    if (!rule.test(elementValue))

                        flag = true;

                } else if (rule instanceof Function) {

                    if (!rule(elementValue))

                        flag = true;

                }

            }

            // great success!

            return flag;

        },

        /**
         * 
         * 사용자가 지정하는 키에 따라 필드를 체크
         * 
         * 특정 필드의 값을 체크하고 싶은 경우 (id, password..)
         * 
         * 키를 지정하고 아래 메소드에 구현하면 된다.
         * 
         * key
         *  - idVerify : 아이디 유효성 체크
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateCustom : function(element, option) {

            var self = this;

            var flag = false;

            var id = self.elementValue(element);

            var elementId = element.attr("id");

            var condition = {};

            switch (option) {

                case "idVerify":

                    condition = {

                        alpha : 1,

                        numeric : 1,

                        start_eng : 1,

                        length : [ 6, 12 ]

                    };

                    break;

                default:

                    break;

            }

            flag = self.validateCustomBiz(element, option, condition);

            if (flag) {

                var labelText = self.errorLabelText(element);

                errorMsg = self.msgProperties(self.rules.custom, labelText, elementId, option);

                // 에러메세지 표시

                self.addErrorText(element, errorMsg);

            }

            // great success!

            return flag;

        },

        /**
         * 
         * password 체크
         * 
         * 
         * 
         * @param pw
         * 
         * @param options
         * 
         * @returns {Boolean}
         * 
         */

        validatePassword : function(element, options) {

            var self = this;

            var flag = false;

            var id = self.elementValue(element);

            var elementId = element.attr("id");

            var userOption = {

                pw1 : {

                    alpha : 1,

                    numeric : 1,

                    length : [ 6, Infinity ]

                },

                pw2 : {

                    alpha : 1,

                    numeric : 1,

                    length : [ 8, Infinity ]

                }

            };

            flag = self.validateCustomBiz(element, options, userOption[options]);

            if (flag) {

                var labelText = self.errorLabelText(element);

                errorMsg = self.msgProperties(self.rules.password, labelText, elementId, options);

                // 에러메세지 표시

                self.addErrorText(element, errorMsg);

            }

            // great success!

            return flag;

        },

        /**
         * 
         * 비교하고 싶은 입력 필드를 지정하면 두 필드의 값이 같은지 체크
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns
         * 
         */

        validateConfirm : function(element, option) {

            var flag;

            var self = this;

            var elementId = element.attr("id");

            var masterEl = $("#" + $.trim(option.split(",")[0]));

            var slaveEl = $("#" + $.trim(option.split(",")[1]));

            if (elementId == masterEl.attr("id")) {

                if (slaveEl.val() != "" && masterEl.val() != slaveEl.val()) {

                    errorMsg = self.msgProperties(self.rules.confirm, labelText, elementId, option);

                    // 에러메세지 표시

                    // self.addErrorText($("#"+slaveEl), errorMsg);

                    self.addErrorText(element, errorMsg);

                    self.removeErrorMsg(slaveEl);

                    flag = true;

                } else {

                    self.removeErrorMsg(slaveEl);

                    flag = false;

                }

            } else { // elementId == slaveId

                if (element.val() != masterEl.val()) {

                    var labelText = self.errorLabelText($(element));

                    errorMsg = self.msgProperties(self.rules.confirm, labelText, elementId, option);

                    // 에러메세지 표시

                    self.addErrorText(element, errorMsg);

                    self.removeErrorMsg(masterEl);

                    flag = true;

                } else {

                    self.removeErrorMsg(masterEl);

                    flag = false;

                }

            }

            return flag;

        },

        /**
         * 
         * 해당 문자의 길이가 또는 값이 x ~ y 사이인지 체크
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateBetween : function(element, option) {

            var flag = false;

            var self = this;

            var len = self.getLength(element, self.elementValue(element));

            var opt = option.split("~");

            var min = Number($.trim(opt[0]));

            var max = Number($.trim(opt[1]));

            var elementId = element.attr("id");

            if (min > max) {

                var tmp = min;

                min = max;

                max = min;

            }

            if (len < min || len > max) {

                var labelText = self.errorLabelText($(element));

                errorMsg = self.msgProperties(self.rules.between, labelText, elementId, option);

                // 에러메세지 표시

                self.addErrorText(element, errorMsg);

                flag = true;

            } else {

                flag = false;

            }

            return flag;

        },

        /**
         * 
         * 기간 입력 필드에 대해 유효성 체크
         * 
         * 특정 기간 범위를 벗어나거나 유효하지 않은 날짜 형식 등을 체크
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateDateTerm : function(element, option) {

            var self = this;

            var flag = false;

            var who = "";

            var startEl = null;

            var endEl = null;

            var range = null;

            var maxRange = null;

            var afterToday = false;

            var options = option.split(",");

            var labelText = " "; // self.errorLabelText($(element));

            var today = $.datepicker.formatDate('yymmdd', new Date());

            var pattern = /^([1-2][0-9][0-9][0-9])([0-1][1-9])([0-3][1-9])$/;

            var form = element.closest(".direct_cld"); // element.closest(".vdForm");

            var label = form.find(".vd-title"); // form.find(".vdTitle");

            if (form.find(".datepicker").length > 0) {

                labelText = self.errorLabelText($(element));

            } else if (label.length > 0) {

                if (label.find("span").length > 0) {

                    labelText = label.html().toUpperCase().split("<SPAN>").join("");

                    labelText = labelText.toUpperCase().split("</SPAN>").join("");

                    labelText = $.trim(labelText);

                    // abelText = label.find("span").html();

                } else {

                    labelText = label.html();

                }

                // console.log(labelText);

            }

            if (options.length < 3) {

                openAlertType1($.hc.messages.get("alert.valid.dateTerm.notfound.range"));

            } else if (options.length == 3) { // start element, end element,
                // date term

                startEl = $("#" + $.trim(options[0]));

                endEl = $("#" + $.trim(options[1]));

                range = "_" + $.trim(options[2]);

            } else if (options.length == 4) { // start element, end element, date term, max range

                startEl = $("#" + $.trim(options[0]));

                endEl = $("#" + $.trim(options[1]));

                range = "_" + $.trim(options[2]);

                maxRange = "_" + $.trim(options[3]);

            } else if (options.length == 5) { // start element, end element, date term, max range, after today

                startEl = $("#" + $.trim(options[0]));

                endEl = $("#" + $.trim(options[1]));

                range = "_" + $.trim(options[2]);

                if ($.trim(options[3]) != "")

                    maxRange = "_" + $.trim(options[3]);

                afterToday = $.trim(options[4]);

            } else if (options.length == 6) { // start element, end element,
                // date term, max range, after today, title

                startEl = $("#" + $.trim(options[0]));

                endEl = $("#" + $.trim(options[1]));

                range = "_" + $.trim(options[2]);

                if ($.trim(options[3]) != "")

                    maxRange = "_" + $.trim(options[3]);

                if ($.trim(options[4]) == "")

                    afterToday = false;

                labelText = $.trim(options[5]) + " " + labelText;

            }

            startEl.val(getYYYYMMDD(startEl.val()));
            endEl.val(getYYYYMMDD(endEl.val()));
            // console.log("start:"+startEl.val()+" end:"+endEl.val());

            /**
             * 
             * 특정일로 부터 날짜를 빼거나 더해 다시 날짜 형식(문자열)으로 반환
             * 
             * @param strDate
             * 
             * @param days
             * 
             * @param idx
             * 
             * @returns
             * 
             */

            function makeDate(strDate, days, idx) {

                var format = /^(\d{4})(\d{2})/;

                var currDate = new Date(strDate.split(format).join("/").replace("/", ""));

                var resultDate = null;

                var diffMonth = null;

                var inputDay = null;

                var lastDay = null;

                var fullYear = null; // 2015-04-09 추가

                if (typeof days == "number") {

                    resultDate = new Date(Date.parse(currDate) - days * 1000 * 60 * 60 * 24);

                } else {

                    fullYear = currDate.getFullYear(); // 2015-04-09 추가

                    diffMonth = parseInt(currDate.getMonth()) - parseInt(days);

                    inputDay = currDate.getDate();

                    // lastDay = (new Date(currDate.getFullYear(), diffMonth, 0)).getDate();

                    lastDay = (new Date(currDate.getFullYear(), parseInt(currDate.getMonth()), 0)).getDate(); // 2015-04-09 // 변경

                    // console.log(idx,"-->",currDate);

                    /*
                     * currDate = new Date();// 2015-04-09 주석 삭제
                     * currDate.setYear(fullYear); // 2015-04-09 추가
                     * currDate.setMonth(diffMonth);// 2015-04-09 위치 변경
                     */
                    currDate = new Date(fullYear, diffMonth, 1);

                    // 2016-03-30 추가 기간검색 오류
                    if (parseInt(diffMonth) == 1) {
                        currDate.setMonth(parseInt(diffMonth));
                    }

                    currDate.setDate(Math.min(inputDay, lastDay));

                    // currDate.setMonth(diffMonth);

                    resultDate = currDate;

                }

                var year = resultDate.getFullYear();

                var month = $.hc.zeroFill(resultDate.getMonth() + 1, 2);

                var day = $.hc.zeroFill(resultDate.getDate(), 2);

                return year + month + day;

            }

            ;

            /**
             * 
             * 날짜 형식이 유효한 지 체크
             * 
             * @param strDate
             * 
             * @returns {Boolean}
             * 
             */

            function dateValidCheck(strDate) {

                var format = /^(\d{4})(\d{2})/;

                var currDate = startEl.val().split(format).join("-").replace("-", "");

                var ret = true;

                try {

                    $.datepicker.parseDate('yy-mm-dd', currDate);

                    if (strDate.length != 8)

                        return false;

                    return true;

                } catch (error) {

                    return false;

                }

            }

            ;

            if (startEl.attr("id") == element.attr("id")) {

                who = "start";

                if (!dateValidCheck(startEl.val())) { // 날짜 형식이 유효한가??

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), {

                        key : "wrongDateType",

                        vars : labelText

                    });

                    self.addErrorText(element, errorMsg);

                    flag = true;

                }

                if (endEl.val() != "" && !flag) {

                    if (!dateValidCheck(endEl.val())) { // 날짜 형식이 유효한가??

                        labelText = self.errorLabelText(endEl);

                        errorMsg = self.msgProperties(self.rules.dateTerm, labelText, endEl.attr("id"), {

                            key : "wrongDateType",

                            vars : labelText

                        });

                        self.addErrorText(endEl, errorMsg);

                        flag = true;

                    }

                }

            } else {

                who = "end";

                if (!dateValidCheck(endEl.val())) { // 날짜 형식이 유효한가??

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), {

                        key : "wrongDateType",

                        vars : labelText

                    });

                    self.addErrorText(element, errorMsg);

                    flag = true;

                }

                if (startEl.val() != "" && !flag) { // 날짜 형식이 유효한가??

                    if (!dateValidCheck(startEl.val())) {

                        labelText = self.errorLabelText(startEl);

                        errorMsg = self.msgProperties(self.rules.dateTerm, labelText, startEl.attr("id"), {

                            key : "wrongDateType",

                            vars : labelText

                        });

                        self.addErrorText(startEl, errorMsg);

                        flag = true;

                    }

                }

            }

            if (endEl.val() != "" && startEl.val() != "" && !flag) { // 검색 종료일과 시작일이 있고 날짜 형식이 유효한가?

                if (startEl.val() < today && afterToday) { // (예약) 시작일이 오늘보다 이전 날짜인가??

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), {

                        key : "startBeforeToday",

                        vars : labelText

                    });

                    self.addErrorText(element, errorMsg);

                    flag = true;

                } else if (endEl.val() < today && afterToday) { // (예약) 종료일이 오늘보다 이전 날짜인가??

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), {

                        key : "endBeforeToday",

                        vars : labelText

                    });

                    self.addErrorText(element, errorMsg);

                    flag = true;

                } else if (startEl.val() > today && !afterToday) { // (검색) 시작일이 오늘보다 이후 날짜인가??

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), {

                        key : "startAfterToday",

                        vars : labelText

                    });

                    self.addErrorText(element, errorMsg);

                    flag = true;

                } else if (endEl.val() > today && !afterToday) { // (검색) 종료일이 오늘보다 이후 날짜인가??

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), {

                        key : "endAfterToday",

                        vars : labelText

                    });

                    self.addErrorText(element, errorMsg);

                    flag = true;

                } else if (startEl.val() > endEl.val()) { // 시작일이 종료일보다 이후
                    // 날짜인가??

                    var obj = {

                        key : "biggerStart",

                        vars : labelText

                    };

                    if (afterToday) {

                        obj.key = "biggerStart.afterToday";

                    }

                    errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), obj);

                    self.addErrorText(element, errorMsg);

                    flag = true;

                } else if (maxRange == null || maxRange == "") { // 최대 검색 일이 없는가?
                    
                    if (makeDate(today, self.dateRange[range].term, 1) > startEl.val()) { // 시작일자가 전체 검색 기간의 시작일보다 작은가??

                        var obj = {

                            key : "overDateTerm",

                            vars : self.dateRange[range].msg

                        };

                        if (afterToday) {

                            obj.key = "overDateTerm.afterToday";

                        }

                        errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), obj);

                        self.addErrorText(element, errorMsg);

                        flag = true;

                    }

                } else if (maxRange != null && maxRange != "") { // 최대 검색 일이 있는가??

                    if (makeDate(today, self.dateRange[maxRange].term, 2) > startEl.val()) { // 시작 일자가 전체 검색 기간의 시작일보다 작은가??

                        var obj = {

                            key : "overMaxDateTerm",

                            vars : self.dateRange[maxRange].msg,

                            vars2 : self.dateRange[range].msg

                        };

                        if (afterToday) {

                            obj.key = "overMaxDateTerm.afterToday";

                        }

                        errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), obj);

                        self.addErrorText(element, errorMsg);

                        flag = true;

                    } else {
                        // console.log(makeDate(endEl.val(),
                        // self.dateRange[range].term, 3), startEl.val());
                        if (makeDate(endEl.val(), self.dateRange[range].term, 3) > startEl.val()) { // 

                            var obj = {

                                key : "overMaxDateTerm",

                                vars : self.dateRange[maxRange].msg,

                                vars2 : self.dateRange[range].msg

                            };

                            if (afterToday) {

                                obj.key = "overMaxDateTerm.afterToday";

                            }

                            errorMsg = self.msgProperties(self.rules.dateTerm, labelText, element.attr("id"), obj);

                            self.addErrorText(element, errorMsg);

                            flag = true;

                        }

                    }

                }

            }

            if (flag) {

                startEl.addClass("error");

                endEl.addClass("error");

            } else {

                endEl.removeClass("error");

                startEl.removeClass("error");

            }

            return flag;

        },

        /**
         * 
         * 입력 필드에 대해 keyup, click, keyup 등과 같은
         * 
         * 특정 이벤트에 실행하고 싶은 유효성 체크
         * 
         * @param element
         * 
         * @param option
         * 
         * @returns {Boolean}
         * 
         */

        validateCustomFunc : function(element, option) {
            var flag = false;
            var self = this;
            var config = self.fixed;
            var elementId = element.attr("id");
            var labelText = self.errorLabelText($(element));
            var events = $.extend(true, {}, $._data($(element)[0], "events"));

            if (option.indexOf(".") >= 0) {
                fn = eval("window." + option);
                $.each(events, function(i, obj) {
                    $.each(obj, function(i, ev) {
                        if (ev.type == "keyup" && ev.namespace.indexOf("jqueryValidate") >= 0) {
                            config.customFuncReturn = fn(ev, element);
                        }
                    });
                });
            } else {
                $.each(events, function(i, obj) {
                    $.each(obj, function(i, ev) {
                        if (ev.type == "keyup" && ev.namespace.indexOf("jqueryValidate") >= 0) {
                            config.customFuncReturn = window[option](ev, element); // 문자
                        }
                    });
                });
            }
            if (null != config.customFuncReturn) {
                errorMsg = self.msgProperties(self.rules.customFunc, labelText, elementId, config.customFuncReturn);
                if (null != config.customFuncReturn.result) {
                    errorMsg = self.msgProperties(self.rules.customFunc, labelText, elementId, config.customFuncReturn.key);
                }
                self.addErrorText(element, errorMsg);
                flag = true;
            } else {
                self.removeErrorMsg(element);
                flag = false;
            }
            return flag;
        },

        /**
         * 날짜 유효성 체크의 경우 
         * 날짜가 유효한가? 
         * 검색 시작일이 오늘날짜보다 이전인가?
         * 검색 시작일과 종료일은 올바르게 입력되었는가? 
         * 검색 기간이 특정 범위(3개월, 6개월 이내만 검색)을 벗어나지 않았는가?
         * 등등 비교할 내용이 많기 때문에 각각의 내용에 대응하는 메시지를 출력한다. 
         *
         * @param name
         * @param label
         * @param option
         * @returns
         */

        getDateTermMessage : function(name, label, option) {
            // console.log("2 - ", name , " : " , label , ":" , option );
            var self = this;
            var maxRange = null;
            var range = null;

            if (typeof label == "undefined" || label == null)
                label = $.trim(label);

            if (name.indexOf("over") != -1) {
                var opt = option.split(",");
                if (opt.length > 3) {
                    range = "_" + $.trim(opt[2]);
                    if ($.trim(opt[3]) != "") maxRange = "_" + $.trim(opt[3]);
                    if (opt.length == 6) label = $.trim(opt[5]) + " " + label;
                    if ($.trim(opt[3]) != "") {
                        message = self.fixed.msgProp.get(name, label, self.dateRange[maxRange].msg, self.dateRange[range].msg);
                    } else {
                        message = self.fixed.msgProp.get(name, label, self.dateRange[range].msg);
                    }
                } else {
                    range = "_" + $.trim(opt[2]);
                    message = self.fixed.msgProp.get(name, label, self.dateRange[range].msg);
                }
            } else {
                message = self.fixed.msgProp.get(name, label, option);
            }
            return message;
        },

        /**
         * error text 화면에 추가
         * @param element
         * @param errorMsg
         */

        addErrorText : function(element, errorMsg) {
            if (!this.notError) {
                return;
            }
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var jsonValid = self.jsonReplace(element.attr(fixed.validAttr));
            var validObj = $.parseJSON(jsonValid);
            //console.log("addErrorText pop:" +  validObj.msgGb +", element.attr(fixed.validAttr):"+element.attr('id'));
            
            // 팝업형태인경우 실행 취소
            if (validObj.msgGb == "pop") {
                return;
            }

            var errorViewTarget = element.siblings();
            var elemContainer = element.closest(".input_cell_box");
            var elemWrapper = element.closest("[class^='box_input']");

            var targetvdMsg = element.closest("[class^='box_input']").find(".vdMsg");
            if (targetvdMsg.hasClass("vdMsg")) {
                elemContainer = element.closest(".vdMsg");
                elemWrapper = element.closest("[class^='box_input']");
            }
            
            if (element.attr("type") == "checkbox") {
                elemContainer = element.closest("[class*='box_chk']");
                elemWrapper = element.closest(".box_terms_wrap");
                if (!element.closest('.terms').siblings('.terms:last').next('.argeeErrorMsg').size()) {
                    element.closest('.terms').siblings('.terms:last').after($('<div class="mt12 argeeErrorMsg"></div>'));
                }
            }

            if (config.initMsgFlg) {
                errorMsg = '';
            }
            // 에러메세지 표시
            if (errorMsg !== '') {
                var errorFlg = true;
                // 형제 노드에 없을시 부모 노드에서 찾아서 추가
                if (errorFlg) {
                    var messageArea = elemWrapper.find(".p3_m_lt_1ln:visible");
                    if (messageArea.length > 0) {
                        var originText = messageArea.html();
                        if (!messageArea.attr("data-origin-text") && !messageArea.hasClass("msg_error")) {
                            messageArea.attr("data-origin-text", originText).html(errorMsg).addClass("msg_error");
                        } else if (!messageArea.attr("data-origin-text") && messageArea.hasClass("msg_error")) {
                            messageArea.html(errorMsg).addClass("msg_error");
                        } else {
                            messageArea.attr("data-origin-text", originText).html(errorMsg);
                        }
                    } else {
                        var addErrorTag = $('<p class="p3_m_lt_1ln msg_error"></p>').html(errorMsg);
                        elemWrapper.append(addErrorTag);
                        if (!elemWrapper.find('.input_cell_box').size()) {
                            addErrorTag.addClass('mt6');
                        }
                    }
                    
                    if (element.attr("type") == "checkbox") {
                        messageArea = $('.argeeErrorMsg');
                        if (messageArea.size()) {
                            if (element.closest('.box_chk').size()) {
                                var addErrorTag = $('<p class="p3_m_lt_1ln fc_m_alert msg_error"></p>').data('errorId', element[0].id).html('&#183;' + errorMsg);
                                messageArea.append(addErrorTag);
                            }
                        }
                    }
                    
                    elemContainer.addClass("error");
                    
                    if (element.closest('.modal_pop').size() && !element.closest('.modal_pop').is(':visible')) {
                        popup.open(element.closest('.modal_pop')[0].id);
                    }
                }
                // hidden field 제어
                if (arguments.length > 2) {
                    var obj = arguments[2];
                    obj.focus();
                } else {

                    // select radio error box 처리 - 2021-06-18 GGCZ05 : 현재 필요는 없으나 추후 삭제 검토
                    if (element.attr("type") == "radio") {
                        element.closest(".radio_box").addClass("error");
                    } else if (element.attr("type") == "checkbox") {
                        elemWrapper.addClass("error");
                    }
                }
                //console.log('%c■■■■ errMsg', 'color: red', self, element, errorMsg);
            }
        },

        /**
         * 에러메세지 전체 삭제 (지정 폼 전체)
         * submit 버튼 클릭시 전체적으로 에러메시지를 삭제하고
         * 다시 유효성 체크를 하여 유효하지 않는 필드에 대해서만 에러를 출력한다.
         */

        removeAllErrorMsg : function() {
            if (!this.notError) {
                return;
            }
            // console.log("removeAllErrorMsg")
            var self = this;
            var fixed = self.fixed;
            /* 2021-06-18 GGCZ05 수정 : 타겟 변경
            self.$elem.find(fixed.errorElement + '.' + fixed.errorClass).remove();
            self.$elem.find(".vdBox" + '.' + fixed.errorClass).removeClass("error");
            self.$validInputs.removeClass("error");
            */
            self.$elem.find(".msg_error").each(function(){
                $(this).html('');
                $(this).closest("[class^='box_input']").find(".input_cell_box").removeClass("error");
                $(this).closest("[class^='box_input']").find(".vdMsg").removeClass("error");
                $(this).closest(".box_terms_wrap").removeClass("error");
                $(this).closest(".box_terms_wrap").find("[class*='box_chk']").removeClass("error");
                // $(this).siblings(".input_cell_box").removeClass("error");
                // $(this).closest(".box_terms_wrap").removeClass("error");
                $(this).closest("[class*='box_chk']").removeClass("error");

                if ($(this).attr("data-origin-text")) {
                    $(this).html($(this).attr("data-origin-text")).removeClass("msg_error");
                } else {
                    $(this).remove();
                }

            });
            $('.argeeErrorMsg').remove();
            self.fixed.firstErrorMsg = '';
        },

        /**
         * 에러메세지 삭제
         * 입력 필드의 keyup 이벤트 발생시 해당 필드의 에러메시지만 삭제하는 경우
         * @param element
         */

        removeErrorMsg : function(element) {
            if (!this.notError) {
                return;
            }
            // console.log("removeErrorMsg")
            var self = this;
            var fixed = self.fixed;
            var config = self.config;
            var errorViewTarget =  element.closest("[class^='box_input']").find(".msg_error");
            if(element.attr("type") == "checkbox"){
                errorViewTarget = element.closest(".box_terms_wrap").find(".msg_error");
            }
            var removeFlg = true;
            $.each(errorViewTarget, function(idx, obj) {
                if ($(this).hasClass("msg_error")) {
                    $(this).html('');
                    if ($(this).attr("data-origin-text")) {
                        $(this).html($(this).attr("data-origin-text")).removeClass("msg_error").removeAttr("data-origin-text");
                    } else {
                        $(this).remove();
                    }
                    removeFlg = false;
                }

            });
            
            if (element.attr("type") == "radio") {
                element.closest(".radio_box").removeClass("error");
            } else if (element.attr("type") == "checkbox") { // 2021-06-18 GGCZ05 else if 추가(약관동의 등 체크박스 그룹)
                element.closest("[class*='box_chk']").removeClass(fixed.errorClass);
                element.closest(".box_terms_wrap").removeClass(fixed.errorClass);
                if (element.closest('.box_chk').size() && $('.argeeErrorMsg').size()) {
                    $('.argeeErrorMsg').children().each(function() {
                        if ($(this).data('errorId') === element[0].id) {
                            $(this).remove();
                        }
                    });
                }
            } else {
                element.closest(".input_cell_box").removeClass(fixed.errorClass);
                var targetvdMsg = element.closest("[class^='box_input']").find(".vdMsg");
                if (targetvdMsg.hasClass("vdMsg")) {
                    element.closest(".vdMsg").removeClass(fixed.errorClass);
                }
            }
            // fixed.firstErrorMsg = '';

        },

        /**
         * element type별 value 취득
         * @param element
         * @returns element value
         */

        elementValue : function(element) {
            var type = $(element).attr("type"), val = $(element).val();
            if (type === "radio") {
                return $("input[name='" + $(element).attr("name") + "']:checked").val();
            }

            if (type === "checkbox") {
                return $("input[id='" + $(element).attr("id") + "']:checked").val();
            }

            if (typeof val === "string") {
                return val.replace(/\r/g, "");
            }
            return val.trim();
        },

        isValueNotNull : function(value) {
            return (value == null || value == '') ? false : true;
        },

        /**
         * element type별 value length 취득
         * @param element
         * @param value
         * @returns
         */

        getLength : function(element, value) {
            //console.log("getLength : " + element[0].nodeName.toLowerCase());
            switch (element[0].nodeName.toLowerCase()) {
                case "select":
                    return $("option:selected", element).length;
                case "input":
                    if ((/radio|checkbox/i).test(element[0].type)) {
                        //console.log(element[0].name,this.$elem.find(':input[name='+element[0].name+']').filter(":checked").length);
                        return this.$elem.find(':input[name=' + element[0].name + ']').filter(":checked").length;
                    }
            }
            return value.length;
        },

        jsonReplace : function(json, word) {
            if (json) {
                return json.split("\\").join("");
            } else {
                return '{}';
            }
        },

        getOnlyNumber : function(value) {
            return value.replace(/[^0-9]/g, '');
        }

    };

    JqueryValidate.defaults = JqueryValidate.prototype.defaults;

    $.fn.hcValidate = function(options, callback) {
        var validate = new JqueryValidate(this, options).init();
        if(typeof validate != 'object'){
            console.log("not validate options : "+validate);
            return null;
        }        
        validate.allErrorCheck(false);
        this.data('validate', validate);
        return validate;
    };

})(jQuery, window, document);

/**
 * 
 * 날짜 형식을 YYYYMMDD 로 변경 (21. 5. 10 -> 20210510)
 * 
 * @param inDateStr
 * @returns
 * 
 */
function getYYYYMMDD(inDateStr) {
    inDateStr = $.trim(inDateStr);
    if (inDateStr.length < 6) {
        return inDateStr;
    }
    // 구분자 . - /
    var splitItem = ".";
    var splitArr = inDateStr.split(splitItem);
    if (splitArr.length > 2) {
        var inYYYY = $.trim(splitArr[0]);
        var inMM = $.trim(splitArr[1]);
        var inDD = $.trim(splitArr[2]);
        if (inYYYY.length == 2) {
            if (inYYYY.substring(0, 1) == "9") {
                inYYYY = "19" + inYYYY;
            } else {
                inYYYY = "20" + inYYYY;
            }
        }
        if (inMM.length == 1) {
            inMM = "0" + inMM;
        }
        if (inDD.length == 1) {
            inDD = "0" + inDD;
        }
        return inYYYY + inMM + inDD;
    } else {
        return inDateStr;
    }
}