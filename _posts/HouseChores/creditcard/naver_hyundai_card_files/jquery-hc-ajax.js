(function($, window, document, undefined){
/**
 * Usage ex) selector를 사용하는 경우 selector는 A, IMG 태그에 한정한다. 
 * $(selector).ajaxCall({ options })
 * 또는
 * $.ajaxCall({ options })
 */
    var submitUrls = [];
	submitUrls.contains = function(item) {
		if (typeof item == "undefined" || !item) return false;
		for (var i=0;i<this.length;i++) 
			if (item == this[i]) return true;
		return false;
	};
	submitUrls.remove = function(item) {
		if (typeof item == "undefined" || !item) return;
		for (var i=0;i<this.length;i++) {
			if (item == this[i]) 
				this.splice(i, 1);
		}
	};
    
    var ajaxCall = function(elem, options) {
        this.$elem = {};
        this.applyElem = {};
        this.options = {};
        this.request = null;
        
        // selector가 없는 경우 - $.ajaxCall({})
        if(elem.selector != ""){
            this.$elem = $(elem);
        }
        this.options = options;
        
        if(!$._isEmptyVal(this.options.applyId)) this.applyElem = $("#"+this.options.applyId);
    };

    ajaxCall.prototype = {
        defaults: {
            _thisData:{},           //외부에서 넘어온 데이터 object
            category:"form",        //Ajax 구분 : form, group, data, load (필수지정)
            applyId:'',             //Ajax 전송 : form, group id  (필수지정)
                                    //단순 load : 페이지를 끼워넣을 element id (필수지정)
            url:'',                 //send url (a tag: 클릭 이벤트시 href의 값으로 대체 / form: action 값으로 대체 가능)
            type:'post',            //send method
            data:{},                //data set - 특정 데이터를 넘기고 싶을 때 사용할 수 있고 기존 필드값이 있으면 merge된다.
            dataType:'json',        //return data type
            async:true,
            showLoading:false,
            success:'',             //success callback
            error:'',               //error callback
            secure : false
        },

        init:/**
         * hcAjax 초기 셋팅
         * 메시지, 사용자 정의 변수 설정 후 ajax 호출
         * @returns {this}
         */
        function() {
            var self = this;
            $.hc.messages.setProperties({name:"ajax-messages"});
            self.config = $.extend({}, self.defaults, self.options);
            self.ajaxSend();
            return this;
        },
        
        
        _loadFnc:        
        /**
         * 단순히 페이지를 로드하기 위해 사용한다.
         * option.category가 load인 경우 내부적으로 호출된다.
         * 
         * $(selector).hcAjax({category:"load", ....})
         */
        function() {
            var self = this;
            if(typeof self.config.success == "function") {
                $.get(self.config.url, function(data){
                    self.config.success(data);
                });

            } else {
                if(self.applyElem.length > 0) {
                    $.get(self.config.url, function(data){
                        self.applyElem.html(data);
                    });
                } else {
                    $.get(self.config.url, function(data){
                       alert(data);
                    });
                }
            }
        },

        
        _ajaxFnc:
        /**
         * option.category가 form, group(div id), data 인 경우 호출되는 내부함수 
         * input의 name을 binding 하여 서버로 전송한다.
         * 
         * 결과처리 
         *  - Ajax 성공
         *      -> 업무 정상 
         *          + 사용자 지정 success function
         *          + 사용자 지정 function이 없을 경우 Default success function 
         *      -> 업무 에러
         *          + 사용자 지정 error function
         *          + 사용자 지정 function이 없을 경우 Default success function
         *  - Ajax 실패
         *      -> 통신 에러
         *          + 사용자 지정 error function
         *          + 사용자 지정 function이 없을 경우 Default success function
         */
        function(){
            var self = this;
            var data = {};
            //if(self.config.data == {}) {
            if(self.config.category != "data") {
                if(self.applyElem.length > 0) {
                    if(self.config.secure) {
                        //makeEncData($("#"+self.config.applyId).get(0));
                    } else {
                        if(typeof hasTranskeyFlag != "undefined") {
                            //console.log("암호화된 값이 넘어가지 않나요?? 옵션에 secure 값을 true로 주세요.");
                        }
                    }
                    
                    $("#"+self.config.applyId+" :input[type != button]").each(function(i, obj){
                        // type이 checkbox
                        
                        if($(this).attr("type") == "checkbox"){
                            // 선택되어 있으면
                            if($(this).is(":checked")) {
                                // 등록된 값이 없다면
                                if(typeof data[$(this).attr("name")] == "undefined" || data[$(this).attr("name")] == "") {
                                    data[$(this).attr("name")] = $(this).val();
                                }else{
                                    data[$(this).attr("name")] += ","+$(this).val();   
                                } 
                            }
                        // type이 radio
                        } else if($(this).attr("type") == "radio") {
                            // 선택되어 있으면
                            if($(this).is(":checked")) {
                                data[$(this).attr("name")] = $(this).val();
                            }
                        } else if($(this).prop("tagName").toUpperCase() == "INPUT") {
                            data[$(this).attr("name")] = $(this).val();
                        // type이 input, text area, select etc
                        } else if($(this).prop("tagName").toUpperCase() == "SELECT" || $(this).prop("tagName").toUpperCase() == "TEXTAREA") {
                            data[$(this).attr("name")] = $(this).val();
                        } 
                        self.config.data = $.extend({}, data, self.config.data);
                    });
                } else {
                    alert($.hc.messages.get("alert.ajax.applyid.missing"));
                    return;
                }
            }
            
            /*
             * 특정 Device에서 App 사용시 AJAX 처리가 SSO 인증이 되지 않는 경우 
             * ssoToken값을 쿠키로 부터 가져와서 서버로 전달할 데이터에 첨가한다.
             * 
             * 2013/09/03 아직까지 이 부분에 대한 처리가 확정되지 않았음    
             */
            /*
            var ssoToken = $.hc.getCookie("ssoToken");
            if( ssoToken != "") {
                self.config.data.ssoToken = ssoToken;
            }
            */
            console.log(submitUrls);
            if ("/cpa/py/apiCPAPY0107_03DT.hc" != self.config.url) {  //20211220 , LBW , 웹리뉴얼에서 건별 예약 내역 중복 호출 가능
                submitUrls.push(self.config.url); 
            }  else if ("/cpp/bm/CPPBM0203_02.hc" != self.config.url) {  //181101 가맹점 적립률 조회는 중복호출 가능  
                submitUrls.push(self.config.url); 
            } else if (submitUrls.contains(self.config.url)) {  //중복 전송 방지 (서버에서 처리)
                if((navigator.userAgent.indexOf("langEN") != -1) || (navigator.userAgent.indexOf("langEn") != -1)) {
                    alert("Still in progress. Please wait");
                } else {
                    alert("처리 중입니다.\n잠시 기다려 주십시오.");
                }
                return;
            } 
            
			this.request = $.ajax({
				type:self.config.type,
				showLoading:self.config.showLoading,
				url:self.config.url,
				data:self.config.data,
				async:self.config.async,
				dataType:self.config.dataType,
				success:
					function(data, textStatus, xhr){
				        if(data.bdy != null && !$._isEmptyVal(data.bdy)  ) {
				            data = data.bdy;
				        }
						// 업무 에러인 경우
						if(!$._isEmptyVal(data.error_code) && !$._isEmptyVal(data.error_message)) {
							// 사용자가 error callback을 만들었을 경우
							if(typeof self.config.error == "function") {
								self.config.error(data.error_code, data.error_message, data, self.config._thisData, textStatus, xhr);
								return;
							// 사용자가 error callback을 만들지 않았을 경우
							} else {
								alert("안내 : " + data.error_message);
								return
							}
							// 사용자 success callback
						} else {
						   // 사용자가 success callback을 만들었을 경우
							if(typeof self.config.success == "function") {
								self.config.success(data, self.config._thisData, textStatus, xhr);
								return;
							}
							// 사용자가 success callback을 만들지 않았을 경우
							else{
								// console.log("success");
								return;
							}
						}
					},
				error:
					function(xhr, textStatus, thrownError) {
						// 서버와 통신이 실패했을 경우

						// 에러 페이지를 설정한 경우 - user callback
						if(typeof self.config.error == "function") {
							self.config.error("안내", $.hc.messages.get("alert.ajax.connect.error.msg"), self.config._self, self.config._thisData, textStatus, xhr);
							return;
						}
						// 에러 페이지가 설정되지 않은 경우
						else {
							alert($.hc.messages.get("alert.ajax.connect.error.codeMsg"));
							return;
						}
					},
				 beforeSend: 
					 function () {
						 if(self.applyElem.length > 0){
							 //alert("start Ajax");
							 //self.applyElem.prepend("<div name='load' id='laod'>loading~~~~</div>");
							 if(self.config.showLoading) HDC.showLoading();
						 }
					 },
				 complete: 
					  function () {
						 if(self.applyElem.length > 0){
							 //alert("end Ajax");
							 //self.applyElem.find("[name=load]").remove();
							 //console.log(self.config.showLoading);
							 HDC.hideLoading();
						 }
						 submitUrls.remove(self.config.url);
						 console.log(submitUrls);
						 this.request = null;
					  }
			});
        },

        ajaxSend:
         /**
         *  option.category의 값으로 페이지를 로드할 것인지 서버로 데이터를 전송할 것인지 결정한다 
         *  option.url이 없는 경우 A link는 href, form은 action 값으로 대치한다.
         *          
         */
        function() {
            var self = this;
            var data = {};
            // selector 가 있는 경우
            if(!$._isEmptyVal(self.$elem.selector)) {
                self.$elem.on("click.ajax", function(e){
                    e.preventDefault();
                    e.stopPropagation();
                // $(this) == self.$elem
                    // A tag 이면 href의 경로를 가져온다.
                    if($(this)._getTagName() == "A" && $._isEmptyVal(self.config.url)) {
                        self.config.url = $(this).attr("href");
                    // form 이면 action의 경로를 가져온다.
                    }else if(self.config.category == "form" && $._isEmptyVal(self.config.url)){
                        self.config.url = self.applyElem.attr("action");
                    }
                    if($._isEmptyVal(self.config.url)) {
                        alert($.hc.messages.get("alert.ajax.url.notFound"));
                    } else {
                        (self.config.category == "load")? self._loadFnc() : self._ajaxFnc();
                    }
               });
            // selector 가 없는 경우
            } else {
              (self.config.category == "load")? self._loadFnc() : self._ajaxFnc();
            }
            return;
        }
    };

    ajaxCall.defaults = ajaxCall.prototype.defaults;

    $.fn.extend({
        _getTagName:function() {
            return $(this)[0].tagName;
        },
        _getMessage:function(key) {

        }
    });

    $.extend({
        _isEmptyVal:function(o) {
            if(typeof o == "undefined" || $.trim(o) == "") return true;
            else if( typeof o == "object" && o == {}) return true;
            else return false;
        }
    });

    $.fn.hcAjax = function(options) {
        var ajaxCallReference = new ajaxCall(this, options);
        return ajaxCallReference.init();
    };

    $.hcAjax = function(options) {
        $.fn.hcAjax(options);
    };

})(jQuery, window, document);