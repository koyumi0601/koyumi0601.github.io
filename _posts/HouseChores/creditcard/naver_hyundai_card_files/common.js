﻿﻿﻿﻿﻿﻿;(function($){$.browserTest=function(a,z){var u='unknown',x='X',m=function(r,h){for(var i=0;i<h.length;i=i+1){r=r.replace(h[i][0],h[i][1]);}return r;},c=function(i,a,b,c){var r={name:m((a.exec(i)||[u,u])[1],b)};r[r.name]=true;r.version=(c.exec(i)||[x,x,x,x])[3];if(r.name.match(/safari/)&&r.version>400){r.version='2.0';}if(r.name==='presto'){r.version=($.browser.version>9.27)?'futhark':'linear_b';}r.versionNumber=parseFloat(r.version,10)||0;r.versionX=(r.version!==x)?(r.version+'').substr(0,1):x;r.className=r.name+r.versionX;return r;};a=(a.match(/Opera|Navigator|Minefield|KHTML|Chrome/)?m(a,[[/(Firefox|MSIE|KHTML,\slike\sGecko|Konqueror)/,''],['Chrome Safari','Chrome'],['KHTML','Konqueror'],['Minefield','Firefox'],['Navigator','Netscape']]):a).toLowerCase();$.browser=$.extend((!z)?$.browser:{},c(a,/(camino|chrome|firefox|netscape|konqueror|lynx|msie|opera|safari)/,[],/(camino|chrome|firefox|netscape|netscape6|opera|version|konqueror|lynx|msie|safari)(\/|\s)([a-z0-9\.\+]*?)(\;|dev|rel|\s|$)/));$.layout=c(a,/(gecko|konqueror|msie|opera|webkit)/,[['konqueror','khtml'],['msie','trident'],['opera','presto']],/(applewebkit|rv|konqueror|msie)(\:|\/|\s)([a-z0-9\.]*?)(\;|\)|\s)/);$.os={name:(/(win|mac|linux|sunos|solaris|iphone)/.exec(navigator.platform.toLowerCase())||[u])[0].replace('sunos','solaris')};if(!z){$('html').addClass([$.os.name,$.browser.name,$.browser.className,$.layout.name,$.layout.className].join(' '));}};$.browserTest(navigator.userAgent);})(jQuery);//http://jquery.thewikies.com/browser/
;(function($){$.fn.bgIframe=$.fn.bgiframe=function(s){if($.browser.msie&&/6.0/.test(navigator.userAgent)){s=$.extend({top:'auto',left:'auto',width:'auto',height:'auto',opacity:true,src:'javascript:false;'},s||{});var prop=function(n){return n&&n.constructor==Number?n+'px':n;},html='<iframe class="bgiframe"frameborder="0"tabindex="-1"src="'+s.src+'"'+'style="display:block;position:absolute;z-index:-1;'+(s.opacity!==false?'filter:Alpha(Opacity=\'0\');':'')+'top:'+(s.top=='auto'?'expression(((parseInt(this.parentNode.currentStyle.borderTopWidth)||0)*-1)+\'px\')':prop(s.top))+';'+'left:'+(s.left=='auto'?'expression(((parseInt(this.parentNode.currentStyle.borderLeftWidth)||0)*-1)+\'px\')':prop(s.left))+';'+'width:'+(s.width=='auto'?'expression(this.parentNode.offsetWidth+\'px\')':prop(s.width))+';'+'height:'+(s.height=='auto'?'expression(this.parentNode.offsetHeight+\'px\')':prop(s.height))+';'+'"/>';return this.each(function(){if($('> iframe.bgiframe',this).length==0)this.insertBefore(document.createElement(html),this.firstChild);});}return this;};})(jQuery);

if(!this.console) { this.console = function() {}; this.console.log = function(str) {}; }


;(function($) {

	/***********************************************
	* 온로드 실행 함수
	************************************************/
	$(function() {

          /**
           * DB 약관동의 조회 팝업
           */
          $("[data-dbagree^=clauses_]").on("click",  function() {
              var $this = $(this);
              _class = $this.data().dbagree;
              _id = $this.attr("id");
              if(_class == null || _class == "") return false;
              var cdList = _class.split("clauses_");
              if (cdList.length >= 2){
                  com_clausesLayer(_class, _id); //약관 레이어 호출
                  return false;
              }      
          });    
          
          $("body").attr({
              ondragstart : 'return false'
          })

	});

})(jQuery); 

/***********************************************
* dataProcessing
************************************************/
function dataProcessing(data) {
	data = data.replace(/<script.*>.*<\/script>/ig,""); // Remove script tags
	data = data.replace(/<\!--[.*>.*<\![endif]-->/ig,"");
	data = data.replace(/<\/?meta.*>/ig,""); //Remove link tags
	data = data.replace(/<\/?link.*>/ig,""); //Remove link tags
	data = data.replace(/<\/?html.*>/ig,""); //Remove html tag
	data = data.replace(/<\/?body.*>/ig,""); //Remove body tag
	data = data.replace(/<\/?head.*>/ig,""); //Remove head tag
	data = data.replace(/<\/?!doctype.*>/ig,""); //Remove doctype
	data = data.replace(/<title.*>.*<\/title>/ig,""); // Remove title tags
	return data;
}


var gbrArr = []; // GNB 롤링 배너 배열 - 2019-09-10 추가

if(!this.console) { this.console = function() {}; this.console.log = function(str) {}; }

var OP_GUBUN = (function(){
        var flag = "P";
        if(document.location.href.indexOf("www.hyundaicard.com") > -1){
                flag = "P";
        } else if(document.location.href.indexOf("tww.hyundaicard.com") > -1){
                flag = "T";
        } else if(document.location.href.indexOf("trewww.hyundaicard.com") > -1){
                flag = "T";                
        } else if(document.location.href.indexOf("btw.hyundaicard.com") > -1){
                flag = "BP";
        } else if(document.location.href.indexOf("twww7.hyundaicard.com") > -1){
                flag = "D";
        } else if(document.location.href.indexOf("drewww.hyundaicard.com") > -1){
                flag = "D";
        } else if (document.location.href.indexOf("localhost") > -1){
                flag = "D";
        }

        return flag;
})();

$(document).ready(function(){
    HDC.initLoading();
});

var HDC = {};

HDC.isUnload = false;
HDC.showCount = 0;
HDC.DEF_TOPPOS = -200;

HDC.unload = function() {
        HDC.showLoading();
};

HDC.screen = {
        unlock: function() {
                var $scope = $("#lock-modal");
                if ($scope.length) {
                        $scope.remove();
                }
        }
};

HDC.initLoading = function() {
        var $scope = $("body");
        var $modal = $("#loading-modal");
        if ($modal.length == 0) {
                $scope.append('<div id="loading-modal" style="display: none; position: fixed; z-index: 10000; top: 0px; left: 0px; height: 100%; width: 100%; opacity: 0.6; background: url(\'data:image/gif;base64,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\') 50% 50% no-repeat rgb(255, 255, 255);filter:alpha(opacity=60)"></div>');
        }
        HDC.showCount = 0;
};

HDC.showLoading = function() {
        //console.log("HDC.showCount+ >> " + HDC.showCount);
        if (HDC.showCount++ > 0) return;

        var $modal = $("#loading-modal");
        $modal.show();
}

HDC.hideLoading = function() {
        HDC.showCount--;
        //console.log("HDC.hidewCount >> " + HDC.showCount);
        if (HDC.showCount > 0) return;

        $("#loading-modal").hide();
        HDC.showCount = 0;
};


HDC.moveFocusPosition = function(el) {
        var $el = $(el);
        if (!$el.length) return;
        var scrollPos = $el.offset().top + HDC.DEF_TOPPOS;
        $(window).scrollTop(scrollPos);
};

function isCmsContent(url) {
        if (url.indexOf("/cms_content/") != -1) return true;
        return false;
}

$(document).ajaxSend(function(event, request, opt){
        var isShow = opt.showLoading || false;
        if (isShow) HDC.showLoading();
});


$(document).ajaxComplete(function(event, response, opt){
        var isShow = opt.showLoading || false;
        if(isShow) HDC.hideLoading();
});

if (typeof viewImgCL !== 'function') {
        this.viewImgCL = function(){};
}

//다음 포커스 엘리먼트 구하기
function  focusNextElement () {
    //add all elements we want to include in our selection
    var focussableElements = 'a:not([disabled]), button:not([disabled]), input[type=text]:not([disabled]), [tabindex]:not([disabled]):not([tabindex="-1"])';
    if (document.activeElement) {
        var focussable = Array.prototype.filter.call(document.querySelectorAll(focussableElements),
        function (element) {
            //check for visibility while always include the current activeElement
            return element.offsetWidth > 0 || element.offsetHeight > 0 || element === document.activeElement
        });
        var index = focussable.indexOf(document.activeElement);
        if(index > -1) {
        var nextElement = focussable[index + 1] || focussable[0];
        //nextElement.focus();
        return nextElement;
        }
    }
}


// 콤마 찍기
function formatComma(num, p) {
        if (!p) p = 0;  //소숫점 이하 자리수
        var re = /(-?\d+)(\d{3}[,.])/;
        var nums = num.toString().split('.');
        nums[0] += '.';
        while (re.test(nums[0])) {
                nums[0] = nums[0].replace(re, '$1,$2');
        }
        if (nums.length > 1) {
                if (nums[1].length > p) {
                        nums[1] = nums[1].substr(0, p);
                }
                return nums.join('');
        } else return nums[0].split('.')[0];
}
// 콤마 삭제
function stripComma(num) {
        return num.replace(/,/g, '');
}
// ajax (경로, 파일명_확장자, 버튼 삭제 여부)
$.fn.loadAjax = function(route,target,delBtn){
        var $this = $(this),
                _class = target.split('-')[0],
                $target = $('.' + _class).not('button'),
                _target = target.replace(/-/g, '.');
        loadIng();
        $.ajax({
                type : "GET",
                url : route + '/' + _target,
                success:function(html){
                        stopIng();
                        $target.append(html);

                },
                complete : function(){
                        if (typeof(delBtn) != 'undefined') $this.remove();
                }
        })
};



/***
 * Global Functions
 */
function fnSSOClear() {
        $.cookie('ssotoken', '',{expires:-1});
}
function fnGoLinkUseSsoToDomain(target, gubun) {
        var domainName = "";
        if (target == 'mybusiness' && 'D' == gubun) domainName = "https://d-mybusiness.hyundaicard.com";
        else if (target == 'mybusiness' && 'T' == gubun) domainName = "https://t-mybusiness.hyundaicard.com";
        else if (target == 'mybusiness') domainName = "https://mybusiness.hyundaicard.com";
        // 2020-01-09 개발계 M몰 SSO 연동을 위한 링크정보 임시 수정요청
        // else if (target == 'mpointmall' && 'D' == gubun) domainName = "https://re-cmpointmall.hyundaicard.com";//운영계
        // else if (target == 'mpointmall' && 'T' == gubun) domainName = "https://pp-mpointmall.hyundaicard.com";//운영계
        else if (target == 'mpointmall' && 'D' == gubun) domainName = "https://dmpointmall.hyundaicard.com";//개발계
        else if (target == 'mpointmall' && 'T' == gubun) domainName = "https://tmpointmall.hyundaicard.com";//개발계
        else if (target == 'mpointmall') domainName = "https://mpointmall.hyundaicard.com";

        return domainName;
}

function fnWindowSso(url, sToken) {
        var ssoWin = window.open("about:blank", "ssoWindow");
        var frm = document.frmAuth;
        frm.action = url;
        frm.method = "post";
        frm.sToken.value = sToken;
        frm.target = "ssoWindow";
        frm.submit();
}

function fnGoLinkUseSso(target) {
        var serverName = "";
        try {
                var strSsoToken = $.cookie("ssotoken");
                if(strSsoToken == "undefined" || typeof(strSsoToken) == "undefined" || strSsoToken == "NO_TOKEN")
                strSsoToken = "";

                switch(target) {
                case "mpointmall" :
                        if (strSsoToken != "") {
                                fnWindowSso(fnGoLinkUseSsoToDomain(target, OP_GUBUN)+"/ssoGate.do?togo=", strSsoToken);
                        } else {
                                window.open(fnGoLinkUseSsoToDomain(target, OP_GUBUN), "_blank", "");
                        }
                        break;
                case "mybusiness":
                        if (strSsoToken != "") {
                if ('D' == OP_GUBUN){
                    fnWindowSso(fnGoLinkUseSsoToDomain(target, OP_GUBUN) + "/common/skeletonSSO.do", strSsoToken);
                }else{
                    fnWindowSso(fnGoLinkUseSsoToDomain(target, OP_GUBUN) + "/common/ssoGate.do", strSsoToken);
                }
                        } else {
                                window.open(fnGoLinkUseSsoToDomain(target, OP_GUBUN), "_blank", "");
                        }
                        break;
                }
        } catch(e) {
                window.open(fnGoLinkUseSsoToDomain(target, OP_GUBUN), "_blank", "");
        }
}


// image resize...(like background cover)
var setImageCover = function(img,wrap,s) {
var wrapFlag = !!wrap ? true : false;
        img.each( function() {
                if ( !wrapFlag ) { wrap = $(this).parent(); }
                if ( !s ) {
                        if ( typeof this.naturalWidth != 'number' ) {
                                s = {
                                        w: getNatural(this).width,
                                        h: getNatural(this).height
                                };
                        } else {
                                s = {
                                        w: this.naturalWidth,
                                        h: this.naturalHeight
                                }
                        }
                }
                var ww = wrap.outerWidth();
                var wh = wrap.outerHeight();
                var w = 0;
                var h = 0;
                if ( ww>s.w ) {
                        w = ww;
                        h = ww*s.h/s.w;
                } else {
                        w = s.w;
                        h = s.h;
                }
                if ( wh>h ) {
                        h = wh;
                        w = wh*s.w/s.h;
                }
                if ( ww>0 && wh>0 ) {
                        $(this).css({width:w+'px', height:h+'px', 'margin':parseInt((wh-h)/2)+'px 0 0 '+parseInt((ww-w)/2)+'px'});
                } else {
                        $(this).removeAttr('style');
                }
        });
};
var getNatural = function(DOMelement) {
        if ( !DOMelement ) {return {width:0,height:0};}
        var img = document.createElement('IMG');
        img.src = DOMelement.src;
        var wid = DOMelement.naturalWidth || img.width;
        var hit = DOMelement.naturalHeight || img.height;
        return {width: wid, height: hit};
};


// 팝업 공통 - 스크롤 없는 팝업
function winPopup1(popUrl, popName, popWidth, popHeight) {
    var winHieght = $(window).height();
    var winWidth = $(window).width();

    var winX = window.screenLeft;
    var winY = window.screenTop;

    var LeftPosition = winX + (winWidth - popWidth)/2;
    var TopPosition = winY + (winHieght - popHeight)/2;

    window.open(popUrl, popName,"width="+popWidth+"=, height="+popHeight+", top="+TopPosition +",left="+LeftPosition+", resizable=no, scrollbars=no, status=no");
}

// 팝업 공통 - 스크롤 있는 팝업
function winPopup2(popUrl, popName, popWidth, popHeight) {
    var winHieght = $(window).height();
    var winWidth = $(window).width();

    var winX = window.screenLeft;
    var winY = window.screenTop;

    var LeftPosition = winX + (winWidth - popWidth)/2;
    var TopPosition = winY + (winHieght - popHeight)/2;

    window.open(popUrl, popName,"width="+popWidth+"=, height="+popHeight+", top="+TopPosition +",left="+LeftPosition+", resizable=yes, scrollbars=yes, status=no");
}


//쿠키값 조회
function getCookie_hc(c_name) {
    try {
        var i,x,y,cookies=document.cookie.split(";");
        for (i=0;i<cookies.length;i++) {
            x=cookies[i].substr(0,cookies[i].indexOf("="));
            y=cookies[i].substr(cookies[i].indexOf("=")+1);
            x=x.replace(/^\s+|\s+$/g,"");

            if (x==c_name) {
                if (decodeURIComponent) return decodeURIComponent(y);
            }
        }
    } catch (e) {}
}
//쿠키값 설정
function setCookie_hc(c_name,value,exdays) {
    var exdate=new Date();
    exdate.setDate(exdate.getDate() + exdays);
    var c_value=encodeURIComponent(value) + ((exdays==null) ? "" : "; expires="+exdate.toUTCString());
    document.cookie=c_name + "=" + c_value +'; path=/ ';
}



  
var _trans = true,
_conScroll = false,
_lastScroll = 0,
_scrollTop = 0,
_scrollDown = true,
_topBtn = false,
_hgt = $(window).height(),
_curFocus = null,
_curFocusObj = null;

//UI 관련 기본 세팅
function pageRoadDesign(){

    $("input.text:not(.blank)").each(function(){

        if($(this).attr("value")=="") {
            $(this).addClass("blank");
            $(this).parent().removeClass("input-wrap-value");
        } else {
            $(this).parent().addClass("input-wrap-value");
        }

    });

    $('input:checked').prev().addClass('on');    // 최초 체크된 서식 디자인 적용

    uiTextAreaCtrl();
    uiFileCtrl();
    uiDatepicker();
    printPage();
    printDialog();
    uiToolTip();
}

function uiTextAreaCtrl(){
    $(".textarea").bind("focus",function(){
        if($(this).val()==$(this).attr("data-txt")) $(this).val("");
    });
    $(".textarea").bind("blur",function(){
        if($(this).val()=="") $(this).val($(this).attr("data-txt"));
    });
}

/*****************************************
filestyle : 파일업로드 폼 디자인
*****************************************/
function uiFileCtrl(){

$(".file").each(function(){
    if($.browser.name=="firefox"){
        $('label[for="'+$(this).attr("id")+'"]').unbind("click.hcpub").bind("click.hcpub",function(){
            $("#"+$(this).attr("for")).trigger("click");
        });
    }
});

$(".file")
.unbind("change.hcpub").bind("change.hcpub", function() {
    if($(this).val()) $('label[for="'+$(this).attr("id")+'"]').addClass("on");
    else $('label[for="'+$(this).attr("id")+'"]').removeClass("on");
    $(this).next(".file-txt").text($(this).val());
})
.unbind("focus.hcpub mouseover.hcpub").bind("focus.hcpub mouseover.hcpub",function(){ $('label[for="'+$(this).attr("id")+'"]').find(".btn-type1").addClass("btn-type1-focus");})
.unbind("blur.hcpub mouseout.hcpub").bind("blur.hcpub mouseout.hcpub",function(){$('label[for="'+$(this).attr("id")+'"]').find(".btn-type1").removeClass("btn-type1-focus");});
}

function uiDatepicker(){
    $(".input-wrap2").each(function(){
        var $that= $(this);
        var $buttonText = $that.find(">.hd-t").text();
        if($that.find("input.text").length>1){
            $(this).find("input.text").eq(0).datepicker({
                dateFormat : "yymmdd",
                showOn: "button",
                showOtherMonths : true,
                showMonthAfterYear : true,
                showButtonPanel: true,
                yearSuffix : "년",
                dayNamesMin : ['일','월','화','수','목','금','토'],
                monthNames : ['01월','02월','03월','04월','05월','06월','07월','08월','09월','10월','11월','12월'],
                buttonText : $buttonText+" 검색시작일 달력으로 선택<span></span>",
                onClose: function( selectedDate ) {
                    $that.find("input.text").eq(1).datepicker( "option", "minDate", selectedDate );
                    $(this).removeClass("blank");
                }
            });
            widget = $that.datepicker( "widget" );
            //console.log(widget);
            $(this).find( "input.text").eq(1).datepicker({
                dateFormat : "yymmdd",
                showOn: "button",
                showOtherMonths : true,
                showMonthAfterYear : true,
                showButtonPanel: true,
                yearSuffix : "년",
                dayNamesMin : ['일','월','화','수','목','금','토'],
                monthNames : ['01월','02월','03월','04월','05월','06월','07월','08월','09월','10월','11월','12월'],
                buttonText : $buttonText+"검색종료일 달력으로 선택<span></span>",
                onClose: function( selectedDate ) {
                    $that.find("input.text").eq(0).datepicker( "option", "maxDate", selectedDate );
                    $(this).removeClass("blank");
                }
            });
        }else{
            $that.find("input.text").eq(0).datepicker({
                dateFormat : "yymmdd",
                showOn: "button",
                showOtherMonths : true,
                showMonthAfterYear : true,
                showButtonPanel: true,
                yearSuffix : "년",
                dayNamesMin : ['일','월','화','수','목','금','토'],
                monthNames : ['01월','02월','03월','04월','05월','06월','07월','08월','09월','10월','11월','12월'],
                buttonText : $buttonText+" 달력으로 선택<span></span>"
            });
        }
    });
}

/***********************************************
* 레이어팝업 닫기
************************************************/
function modalClose(){
    var modelEventEl = $('a[href="#'+$('.wrap-layer-on').attr('id')+'"]').eq(0);
    $(".wrap-layer-on").remove();
    $(".modal-overlay").fadeOut();
    modelEventEl.focus();
}

/***********************************************
* 인쇄 : 레이어
************************************************/
function printDialog(){
    $(".wrap-layer-on").find(".btn-print").bind("click",function(ev){
        if(window.print){
            $('body').addClass("print-dialog");
            window.print();
            $('body').removeClass("print-dialog");
            modalClose();
        }
        ev.preventDefault();
    });
}

/***********************************************
* 인쇄 : 페이지
************************************************/
function printPage(){

    /*  페이지 인쇄 */
    $(".btn-print").unbind("click.hcpub").bind("click.hcpub",function(ev){
        if(window.print){
            $('body').addClass("print-page");
            window.print();
            $('body').removeClass("print-page");
            //ev.preventDefault();
        }
    });

    /*  약관 인쇄 */
    $(".form-agree .top").each(function(){
        $(this).find(".btn-type4").eq(1).modalCon({
            callbackBefore: setPringDialog,
            callbackLoad: setPrintDialogPage
        });
    });

    function setPringDialog(){
        var agreeFiledset = $(this).parent().parent().parent();
        var agreeContent = "";

        if(agreeFiledset.find(".scroll-content .con-type2").length>0){
            if(agreeFiledset.find(".scroll-container").length>0)  agreeContent = agreeFiledset.find(".scroll-container").eq(0).html();
            else agreeContent = agreeFiledset.find(".scroll-content").eq(0).html();
        }else{
            return ;
        }

        var printLayout='';
        printLayout += '<div class="layer-type1 layer-type1-w3">';
        printLayout += '<div class="layer-header">';
        printLayout += '<h1>'+$(this).parent().parent().find("label").html()+'</h1>';
        printLayout += '<button type="button" class="btn-close">레이어닫기<span></span></button>';
        printLayout += '</div>';
        printLayout += '<div class="layer-content">';
        printLayout += '<div class="scroll-content scroll-border scroll-agree">';
        printLayout += agreeContent;
        printLayout += '</div>';
        printLayout += '<div class="btn-right"><span class="btn-type1"><a href="#none" class="btn-print">인쇄</a></span></div>';
        printLayout += '</div>';
        printLayout += '</div>';
        printLayout += '</div>';
        return printLayout;
    }

    function setPrintDialogPage(){
        $(".wrap-layer-on .scroll-content").jScrollPane();
    }
    //ev.preventDefault();
}

function uiToolTip(){  
    if ($('.tooltip-wrap2').length) {
        $('.tooltip-wrap2').toggleContents({
            event_element:'.btn-type4',
            view_container: '.tooltip2',
            default_display : true,
            animation : {
                effect: 'blind',
                easing: 'easeInOutQuart',
                duration: 0
            },
            callbackBefore : function(){
                var focus = $('.btn-type4', this);
                $(this).find(".btn-close").bind("click",function(){
                    $(focus).focus();
                    $(".tooltip2").hide();
                });
            }
        });
    }
}


//My Account 리스트 펼치기
var detailView = ( function ($wrap, $btn, type) {
        var _$wrap = $wrap,
                _$detailBtn = $wrap.find( $btn ),
                _$detailCont =  $wrap.find( '.spread_fold' );
                _$detailCont2 =  $wrap.find( '.detail_amount' ),
                _cur = $(window).scrollTop(),
                _topGab = 55;
        if ($('.area_top_tit').length) _topGab = 115;

        var _$type = type,
                _crt = 0;

        contOpen();

        function contOpen () {
                $wrap.each( function () {
                        if ( $(this).hasClass( 'view' ) ) {
                                $(this).find( _$detailBtn ).addClass( 'on' );
                                $(this).find( _$detailCont ).show();
                        }
                });

                _$detailBtn.on( 'click', function (e) {
                        if($('.breakdown_inquery_section.fold').length > 0)
                                _topGab = 155;
                        else
                                _topGab = 115;

                        var _offset2 = $(this).offset().top - _topGab;

                        if ( $(this).closest( $wrap ).hasClass( 'view' ) ) {
                                _conScroll = true;
                                $(this).removeClass( 'on' );
                                if ( _$type ) {
                                        // list가 1개일 때
                                        $(this).closest( $wrap ).removeClass( 'view' );
                                        $(this).closest( $wrap ).find( '.spread_fold' ).eq( _crt ).slideUp(300, function(){
                                                setTimeout(function(){
                                                        _conScroll = false;
                                                },100);
                                        });
                                } else {
                                        // list가 여러개일 때
                                        $(this).closest( $wrap ).removeClass( 'view' );
                                        $(this).parent().siblings().slideUp(300,function(){
                                                setTimeout(function(){
                                                        _conScroll = false;
                                                },100);
                                        });
                                }
                                $(this).find( 'span.blind' ).remove();
                                $(this).append( '<span class="blind">펼치기</span>' );
                        } else {
                                $(this).addClass( 'on' );
                                if ( _$type ) {
                                        // list가 1개일 때
                                        $(this).closest( $wrap ).addClass( 'view' );
                                        $(this).closest( $wrap ).find( '.spread_fold' ).eq( _crt ).slideDown(300, function(){
                                                $('body, html').animate({ scrollTop : _offset2 }, 500, 'easeInOutCubic' );
                                        });
                                } else {
                                        // list가 여러개일 때
                                        $(this).closest( $wrap ).addClass( 'view' );
                                        if($(this).parents(".history_list")){ // 카드이용내역일때
                                                $(this).parent().next().slideDown(300);
                                        }else{
                                                $(this).parent().next().slideDown(300, function(){
                                                        $('body, html').animate({ scrollTop : _offset2 }, 500, 'easeInOutCubic' );
                                                });
                                        }
                                }

                                $(this).find( 'span.blind' ).remove();
                                $(this).append( '<span class="blind">접기</span>' );
                        };
                        e.preventDefault();
                        return false;
                });
        }
});
(function($) {
    /***********************************************
    * 레이어
    ************************************************/
    var _add = false;
    var _callbackAfter;

    // 딤드 클릭시 레이어 닫기
    $(document).on('click','.layer_dimmed',function(){
        $('.area_layer.opened').closest('.wrap-layer-on').closeLayer_();
    });
    $.fn.closeLayer_ = function(target){
        var $target = $(this);
        $target.removeClass('opened').fadeOut(function(){
            $target.remove();
            // 레이어가 1개 이상인 경우 처리
            $('.area_layer').css({ 'z-index' : '20' });
            _add = false;
        });
        // 레이어가 1개 이상인 경우 처리
        if (!_add) {
            $('html').removeClass('area_layer-opened');
            $('.dimmed').css({ opacity : '0' },300);
            setTimeout(function(){
                $('.dimmed').remove();
            },300);
        }

        if (typeof _callbackAfter === 'function') {
            _callbackAfter.call();
        };
    };

    $.fn.modalCon = function(options){
        return this.each(function(n) {
            options = options || {};
            var opts = $.extend({}, $.fn.modalCon.defaults, options || {});
            var that = this;
            var $cont = $(this);    //이벤트호출객체 a
            var $contWrap;          //레이어컨텐츠
            var $contCon;           //레이어컨텐츠내부 컨텐츠영역
            var contWrapID;         //레이어 아이디

            $cont.bind('click', function(ev) {
                ev.preventDefault();
                init();
            });

            var init = function() {

                if (opts.callbackBefore) {
                    if (typeof opts.callbackBefore === 'function') {
                        opts.data = opts.callbackBefore.call($cont);
                        if (!opts.data) return;
                    }
                }

                var hasHref = $cont.attr("href");
                if(typeof hasHref != "undefined") {
                    contWrapID = 'layer_' + $cont.attr("href").split('#')[1];
                }
                else {
                    contWrapID = 'layer_default';
                }

                $contWrap = $('<div />', {'class': opts.onClass, 'id': contWrapID}).html(opts.data);

                /* 레이어팝업 컨텐츠 세팅 */
                if (opts.url) { // url trim 검사 필요(추후 trim 범용 함수 제작)
                    ajaxCall(opts.url, $contWrap, opts.sCheck, function() {
                        setModalCon();
                    });
                }

                if (opts.data) {
                    // 레이어가 이미 열려있는 경우 처리
                    if ($('.area_layer:visible').length > 1) {
                        $('#wrap').find('.area_layer').css({ 'z-index' : 10 });
                        _add = true;
                    }

                    $('#wrap').append($contWrap);
                    setModalCon();

                    _callbackAfter = opts.callbackAfter;
                }

                /* 레이어오픈 : 이벤트 및 애니메이션 설정 */
                function setModalCon(){
                    var $layer = $('#wrap').find( '#' + contWrapID);

                    if ($layer.find('select').length){
                        var _size = $layer.find('select').size();
                        for (var i = 0; i < _size; i++){
                            selectAdd($('#' + contWrapID + ' select').eq(i)); // 셀렉트 박스 활성화
                        }
                    }
                    $layer.find('.area_layer').addClass('opened').fadeIn();
                    $('html').addClass('area_layer-opened');
                    if (_trans && _hgt < $layer.height()) $layer.perfectScrollbar();

                    // 특정 클래스에 tabindex 추가
                    if ($('.area_layer .agree_content').length){
                        $('.area_layer .agree_content').attr('tabindex','0');
                        $layer.find('.box_layer > H3').attr('tabindex',-1).focus();
                    } else {
                        $layer.find('input, .btn_select a, a, textarea, button').first().focus();
                    }

                    // 레이어가 이미 있는 경우 처리
                    var $dimm = '<span class="dimmed layer_dimmed"></span>';
                    if (!$('body .dimmed').length || !_add) $('body').prepend($dimm).find('.dimmed').css({ opacity : '1' });

                    $cont.addClass('open_btn');

                    // 센스리더 구동시에는 타켓 지정된 첫 번째 요소로 이동하지만 포커스는 닫기 버튼으로 이동시킨다.
                    $contWrap.find(opts.layerContent).eq(0).focus();
                    
                    // 모달 팝업 띄우기
                    $('#'+contWrapID).find('.modal_pop').addClass("active");
                    $('#'+contWrapID).find('.modal_pop').attr('style','opacity:1');

                    //로드 콜백함수 실행
                    if (typeof opts.callbackLoad === 'function') {
                        opts.callbackLoad.call($layer);
                    };

                    //닫기버튼 이벤트설정
                    $contWrap.find(opts.close_trigger).on("click",function(e){
                        var $this = $(this),
                        _keyCode = e.keyCode || e.which;
                        if (_keyCode == 9){
                            e.preventDefault();
                            $('.open_btn').focus().addClass('opened_tab');
                        } else {
                            $contWrap.closeLayer_();
                            $('.open_btn').focus().addClass('opened_tab');
                            $cont.removeClass('open_btn').removeClass('opened_tab');
                        }

                        /*
                        closeLayer_ function 으로 이동
                        if (typeof opts.callbackAfter === 'function') {
                            opts.callbackAfter.call($cont);
                        };
                        */

                        return false;
                    });
                }
            };
        });
    };

    $.fn.modalCon.defaults = {
        modal : true,
//      modalClass : "modal-overlay",
        modalEffect : true,
        onClass : "wrap-layer-on",
        layerWrap : ".area_layer",
        layerContent : ".box_layer",
        close_trigger : '.btn_close a',
        appendNext : true,
        url: false,
        data : true,
        sCheck : false,
        positionTop : false
//      callbackBefore: null,//function() {},
//      callbackLoad : null,//function() {},
//      callbackAfter: null//function() {},
    }
})(jQuery);


/**
 * 모달 알럿 1
 * @openAlertType1(msg, focusId, callback)
 *
 * - focusId : 버튼 ID (focus 처리용)
 * - msg : alert message
 * - callback function
 */
 function openAlertType1(msg, focusId, afterFn){
    var closeEl   = '.confirm';
    var id        = "layerOpenAlertType1";
    var alertHtml =
    '<div id="'+ id +'" class="modal_pop modal_alert" tabindex="-1" role="dialog" aria-modal="true">' +
        '<div class="modal_wrap">' +
            '<div class="modal_container">' +
                '<div class="layer_wrap w288">' +
                    '<div class="layer_body">' +
                        '<p class="p2_m_2ln fc_m_a64">' + msg + '</p>' +
                    '</div>' +
                    '<div class="layer_footer">' +
                        '<div class="box_btn col1">' +
                            '<a href="javascript:javascript:void(0);" class="btn56_outline02_boldtxt confirm"><span>확인</span></a>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>' +
    '</div>';
    // 한번만 생성
    if($('#'+id).length){
        modal.close(id, '');         
        $('#'+id).empty().remove();
    }
    var $targetSource = $('.content');
    if ( $targetSource.length == 0 ) {
        $targetSource = $('.contents');
    }
    $targetSource.first().after(alertHtml);
    $('#'+id).find(closeEl).on('click', function(){
        modal.close(id, focusId);

        // 알럿 닫힌 후에 실행되어야 할 함수가 있을 경우 실행
        if(afterFn !== undefined){
            afterFn();
        }
    });
        
    if ( $('.nppfs-keypad:visible').length > 0 ) {
        $(".nppfs-keypad").css("display","none"); //열려있는 키패드 숨김처리
    }
    modal.open(id, focusId);
    $('#'+id).attr('tabindex',0).focus();
 }
 
 /**
  * 모달 알럿 2
  * @openAlertType2(msg1, msg2, focusId, callback, btn)
  *
   * - msg1 : message (내용) 
   * - msg2 : message (제목) 
   * - focusId : 버튼 ID (focus 처리용)
   * - callback function
   * - callback function
   * - btn : 확인 버튼명
  */
  function openAlertType2(msg1, msg2, focusId, afterFn, btn){
     var closeEl   = '.confirm';
     var id        = "layerOpenAlertType2";     
     if ( typeof btn == 'undefined' || btn == '') {
         btn = "확인";
     } 
     var alertHtml =
     '<div id="'+ id +'" class="modal_pop modal_alert" tabindex="-1" role="dialog" aria-modal="true">' +
         '<div class="modal_wrap">' +
             '<div class="modal_container">' +
                 '<div class="layer_wrap w288">' +
                     '<div class="layer_body">';
     if (msg2 != '') {                   
         alertHtml +='<p class="p1_b_2ln fc_m_blk">' + msg2 + '</p>';
     }
     alertHtml +=    '<p class="p2_m_2ln fc_m_a64">' + msg1 + '</p>' +
                     '</div>' +
                     '<div class="layer_footer">' +
                         '<div class="box_btn col1">' +
                             '<a href="javascript:javascript:void(0);" class="btn56_outline02_boldtxt confirm"><span>' + btn + '</span></a>' +
                         '</div>' +
                     '</div>' +
                 '</div>' +
             '</div>' +
         '</div>' +
     '</div>';
     // 한번만 생성
     if($('#'+id).length){
         modal.close(id, '');         
         $('#'+id).empty().remove();
     }
     var $targetSource = $('.content');
     if ( $targetSource.length == 0 ) {
         $targetSource = $('.contents');
     }
     $targetSource.first().after(alertHtml);
     $('#'+id).find(closeEl).on('click', function(){
         modal.close(id, focusId);

         // 알럿 닫힌 후에 실행되어야 할 함수가 있을 경우 실행
         if(afterFn !== undefined){
             afterFn();
         }
     });
     if ( $('.nppfs-keypad:visible').length > 0 ) {
         $(".nppfs-keypad").css("display","none"); //열려있는 키패드 숨김처리
     }
     modal.open(id, focusId);
     $('#'+id).attr('tabindex',0).focus();
  }
 
 
 /**
  * 모달 컨펌1 (내용, 확인콜백, 취소콜백)
  * @openConfirmType1(msg, callback1, callback2)
  *
  * - msg : message (내용)
  * - afterFn : callback function(확인)
  * - afterCFn : callback function(취소)
  */
  function openConfirmType1(msg, afterFn, afterCFn){
     var closeEl   = '.cancel';
     var confirmEl   = '.confirm';   
     var id        = "layerOpenConfirmType1";
     var alertHtml =
     '<div id="'+ id +'" class="modal_pop modal_alert" tabindex="-1" role="dialog" aria-modal="true">' +
         '<div class="modal_wrap">' +
             '<div class="modal_container">' +
                 '<div class="layer_wrap w288">' +
                     '<div class="layer_body">' +
                         '<p class="p2_m_2ln fc_m_a64">' + msg + '</p>' +
                     '</div>' +
                     '<div class="layer_footer">' +
                         '<div class="box_btn col2">' +
                             '<a href="javascript:void(0);" class="btn56_outline02_boldtxt cancel"><span class="fc_m_a48">취소</span></a>' +
                             '<a href="javascript:void(0);" class="btn56_outline02_boldtxt confirm"><span>확인</span></a>' +
                         '</div>' +
                     '</div>' +
                 '</div>' +
             '</div>' +
         '</div>' +
     '</div>';
     // 한번만 생성
     if($('#'+id).length){
         $('#'+id).empty().remove();
     }
     var $targetSource = $('.content');
     if ( $targetSource.length == 0 ) {
         $targetSource = $('.contents');
     }
     $targetSource.first().after(alertHtml);
     $('#'+id).find(closeEl).on('click', function(){
         modal.close(id, '');          
         if(afterCFn !== undefined && afterCFn != ''){
             afterCFn();
         }
     });
     $('#'+id).find(confirmEl).on('click', function(){
         modal.close(id, '');
         // 알럿 닫힌 후에 실행되어야 할 함수가 있을 경우 실행
         if(afterFn !== undefined && afterFn != ''){
             afterFn();
         }
     });

     if ( $('.nppfs-keypad:visible').length > 0 ) {
         $(".nppfs-keypad").css("display","none"); //열려있는 키패드 숨김처리
     }
     modal.open(id, '');
     $('#'+id).attr('tabindex',0).focus();
  }
   
  

  /**
   * 모달 컨펌2 확장 (내용+제목, 버튼명 변경, 확인콜백, 취소콜백, 버튼위치, 버튼 강조css위치, 팝업닫기여부)
   * @openConfirmType2(msg1, msg2, btn1, btn2, callback1, callback2, btnPos, cssPos, closeYn)
   *
   * - msg1 : message (내용) 
   * - msg2 : message (제목)
   * - btn1 : 취소 버튼명
   * - btn2 : 동작 버튼명
   * - afterFn : callback function(확인)
   * - afterCfn : callback function(취소)
   * - btnPos : 버튼위치 ('column' : 상하버튼, 그외 : 좌우버튼)  
   * - cssPos : 버튼강조css위치  ('second':버튼회색css-우/하, 그외 : 버튼회색css-좌/상)
   * - closeYn : 팝업닫기 여부 ('N' : 팝업유지 , 그외 : 팝업닫힘)
   */
   function openConfirmType2(msg1, msg2, btn1, btn2, afterFn, afterCFn, btnPos, cssPos, closeYn){
      var closeEl   = '.cancel';
      var confirmEl = '.confirm';   
      var id        = "layerOpenConfirmType2";
      var cssBtn1   = "fc_m_a48";
      var cssBtn2   = "";      
      if (btn1 == '') {
          btn1 = "취소";
      }
      if (btn2 == '') {
          btn2 = "확인";
      }   
      if ( cssPos == "second") {
          cssBtn1   = "";
          cssBtn2   = "fc_m_a48";
      }      
      var btnHtml   = '<div class="box_btn col2">' +
                          '<a href="javascript:void(0);" class="btn56_outline02_boldtxt cancel"><span class="' + cssBtn1 + '">' + btn1 + '</span></a>' +
                          '<a href="javascript:void(0);" class="btn56_outline02_boldtxt confirm"><span class="' + cssBtn2 + '">' + btn2 + '</span></a>' +
                      '</div>';
      if ( btnPos == 'column' ) {    
         btnHtml   = '<div class="box_btn col1">' +
                          '<a href="javascript:void(0);" class="btn56_outline02_boldtxt cancel"><span class="' + cssBtn1 + '">' + btn1 + '</span></a>' +
                      '</div>' +    
                      '<div class="box_btn col1">' +
                          '<a href="javascript:void(0);" class="btn56_outline02_boldtxt confirm"><span class="' + cssBtn2 + '">' + btn2 + '</span></a>' +
                      '</div>';
      } 
      var alertHtml =
      '<div id="'+ id +'" class="modal_pop modal_alert" tabindex="-1" role="dialog" aria-modal="true">' +
          '<div class="modal_wrap">' +
              '<div class="modal_container">' +
                  '<div class="layer_wrap w288">' +
                      '<div class="layer_body">';
      if (msg2 != '') {                   
          alertHtml +='<p class="p1_b_2ln fc_m_blk">' + msg2 + '</p>';
      }
      alertHtml +=    '<p class="p2_m_2ln fc_m_a64">' + msg1 + '</p>' +
                      '</div>' +
                      '<div class="layer_footer">' + 
                       btnHtml +
                      '</div>' +
                  '</div>' +
              '</div>' +
          '</div>' +
      '</div>';
      // 한번만 생성
      if($('#'+id).length){
          modal.close(id, '');           
          $('#'+id).empty().remove();
      }
      var $targetSource = $('.content');
      if ( $targetSource.length == 0 ) {
          $targetSource = $('.contents');
      }
      $targetSource.first().after(alertHtml);
      $('#'+id).find(closeEl).on('click', function(){
          if ( closeYn != "N" ) { modal.close(id, ''); }         
          if(afterCFn !== undefined && afterCFn != ''){
              afterCFn();
          }
      });
      $('#'+id).find(confirmEl).on('click', function(){
          if ( closeYn != "N" ) { modal.close(id, ''); }   
          // 알럿 닫힌 후에 실행되어야 할 함수가 있을 경우 실행
          if(afterFn !== undefined && afterFn != ''){
              afterFn();
          }
      });
      if ( $('.nppfs-keypad:visible').length > 0 ) {
          $(".nppfs-keypad").css("display","none"); //열려있는 키패드 숨김처리
      }
      modal.open(id, '');
      $('#'+id).attr('tabindex',0).focus();
   }
   
   
 
   //DB조회 약관 레이어 호출
   function com_clausesLayer(_cd, _parentId){
           var cd = _cd.replace('clauses_','');
           var id = "comTableAgreePopup";
           $.hcAjax({
                   category:"data",
                   showLoading:true,
                   type : 'get',
                   data: { "cd" : cd, "parentId" : _parentId },
                   url : '/cpe/ec/apiCPEEC0701_01.hc',
                   success : function(data){
                       if (data.result == -1){
                           openAlertType1("해당 약관은 등록되어 있지 않습니다.", _parentId);
                           return false;
                       }
                       if($('#'+id).length){ // 한번만 생성
                           $('#'+id).empty().remove();
                       }                           
                       $('#container').append(data.html);
                       popup.open(id, _parentId);

                   },
                   error : function(code, message, data, textStatus, xhr) {
                       openAlertType1(code+":"+message);
                       return false;
                   },
                   complete : function(){
                       HDC.hideLoading();
                   }
           })
   }   

   

  /**
   * 날짜타입 적용 앱소스 참고하여 작업
   */
  function moment(date, type){
      var viewDate;
          
      var year = date.substring(2,4);
      var month = date.substring(4,6);
      var day = date.substring(6,8);
      if(month.substring(0,1) == '0'){
          month = month.substring(1,2);
      }
      if(day.substring(0,1) == '0'){
          day = day.substring(1,2);
      }
      
      viewDate = year + '. ' + month + '. ' + day;
      
      if(date.length > 8 && type != "DATE_FORMAT"){
          viewDate += ' ・ ' + date.substring(8,10) + ':' + date.substring(10,12);
          if(type == "DATE_FORMAT_TIME_SEC"){
              viewDate = viewDate + ':' + date.substring(12,14);
          }
      }
      if(type == "HH:mm"){
          viewDate = date.substring(8,10) + ':' + date.substring(10,12);
      }else if(type == "YYMM"){
          viewDate = year + '. ' + month;
      }
      return viewDate;
  }

  /**
   * '지금': 1분 미만
   * 'OO분전': 1분 ~ 59분
   * 'O시간 전': 1시간 ~ 3시간
   * '오늘 + 24시간제시간 (13:01)' : 3시간 ~ 결제일 당일
   * '날짜 + 24시간제시간 (19.12.25 ・ 13:01) : 결제일 다음날 부터
   */
   
  function toDaySeting(date){
      var dateString = moment(date,"DATE_FORMAT_TIME_SEC");
      var toDate = "${hc_date:getCurrentDateDefault().substring(0,8)}";
      var serverHour = "${hc_date:getCurrentTimeDefault().substring(0,2)}";
      var serverMinute = "${hc_date:getCurrentTimeDefault().substring(2,4)}";
      
      // 서버응답 일자와 입력된 일자 비교, 같을 경우 '오늘'이라고 표기
      if (date.substring(0,8) == toDate) {
          if (date.length == 8 ) {
              dateString = '오늘';
          } else {
              var viewDate = "";
              if(date.length > 8){

                  var serverTime = Number(serverHour*60) + Number(serverMinute);
                  var searchTime = Number(date.substring(8,10)*60) + Number(date.substring(10,12));
                  var minutes = serverTime - searchTime;
                  console.log( "serverTime : " +serverTime+ " searchTime : " +searchTime+ " minutes : " + minutes);
                  if (minutes < 1) {                                            // '지금': 1분 미만
                      dateString = '지금';
                  } else if (minutes < 60) {                                    // 'OO분전': 1분 ~ 59분
                      dateString =  minutes + '분 전';
                  } else if (minutes < 239) {                                   // 'O시간 전': 1시간 ~ 3시간
                      dateString = Math.floor(minutes / 60) + '시간 전';
                  }else{
                      dateString = '오늘' + viewDate;
                  }
              }
              
          }
      }
      return dateString;
  }   

  function viewDate(searchDt, format){
      var toYear = "${hc_date:getCurrentDateDefault().substring(0,4)}";
      var viewYear = "";
      if(toYear != searchDt.substring(0,4) && format == "년월일"){
          viewYear = searchDt.substring(0,4) + "년 ";
      }
      if(format == "YY. M. d"){
          viewYear = searchDt.substring(2,4) + "년 ";
      }
      var viewMonth = searchDt.substring(4,6) + "월 ";
      if(viewMonth.substring(0,1) == "0"){
          viewMonth = viewMonth.substring(1,searchDt.length);
      }
      var viewDay = "";
      var viewDt = "";
      if(format == "년월"){
          viewDt = viewYear + viewMonth;
      }else if(format == "월"){
          viewDt = viewMonth;
      }else if(format == "월일"){
          viewDay = searchDt.substring(6,8) + "일";
          if(viewDay.substring(0,1) == "0"){
              viewDay = viewDay.substring(1,searchDt.length);
          }
          viewDt = viewMonth + viewDay;
      }else{
          viewDay = searchDt.substring(6,8) + "일";
          if(viewDay.substring(0,1) == "0"){
              viewDay = viewDay.substring(1,searchDt.length);
          }
          viewDt = viewYear + viewMonth + viewDay;
      }
      return viewDt;
  }
  
  
  /**
   * 인증의 확인 버튼 활성화 체크
   * el : 입력되어야 하는 폼
   * btnId : 버튼ID 
   */
  function comEmptyElCheck(el, btnId){
      $(el).on('keydown keyup blur change input focusout', function(){
         var emptyEl = [];
         $(el).each(function(){
             if ($(this).attr("type") == "checkbox") {
                 if(!$(this).is(":checked")){
                     emptyEl.push($(this));
                 }
             } else {
                if($(this).val() == ''){
                    emptyEl.push($(this));
                }
             }       
      
         });
         var isValAll = emptyEl.length === 0;
         $(btnId).prop('disabled', isValAll ? false : true);
      });
  }      

  /**
   * 이미지 Path 정제 작업
   * @param pathNm : 이미지 URL
   * @param viewNm : 이미지 View 결로
   * @returns {String}
   */
  function imgUrlRefn(pathNm, viewNm) {
      return pathNm.replace(new RegExp('.+\/(.+)|.+'), function() {
          if(arguments[1]){
              return arguments[0].replace(new RegExp('.+(webdoc)'), '');
          } else {
              return /\/$/.test(viewNm) ? viewNm + arguments[0] : viewNm + '/' + arguments[0];
          }
      });
  }
  
  /**
   * 한글입력만 리턴
   */
  function comHangleChk(value){
      var spChar = /[\\~`!@\#|$%^&*\()\=+_,'"{}\[\]:\;\/<>?\s.0-9a-z]/gi;
      if (spChar.test(value)) {
          value = value.replace(spChar, '');
          return value;
      }    
      return value;
  }  
  
  /**
   * 한글/영문입력만 리턴 (영문은 대문자 변환)
   */
  function comHangleEngChk(value){
      var spChar = /[\\~`!@\#|$%^&*\()\=+_,'"{}\[\]:\;\/<>?.0-9]/gi;
      if (spChar.test(value)) {
          value = value.replace(spChar, '');
          return value.toUpperCase();
      }    
      return value.toUpperCase();
  }    
  
  /**
   * 한글입력 체크
   */
  function comIsHangleChk(value){
      var spChar = /[\\~`!@\#|$%^&*\()\=+_,'"{}\[\]:\;\/<>?\s.0-9a-z]/gi;
      if (spChar.test(value)) {
          return false;
      }    
      return true;
  }  
  
  
  /**
   * 한글/영문입력 체크(공백 포함)
   */
  function comIsHangleEngChk(value){
      var spChar = /[\\~`!@\#|$%^&*\()\=+_,'"{}\[\]:\;\/<>?.0-9]/gi;
      if (spChar.test(value)) {
          return false;
      }    
      return true;
  }    
  
  
  /**
   * 숫자입력만 리턴
   */
  function comOnlyNumerChk(value){
      var spChar = /[^0-9]/gi;
      if (spChar.test(value)) {
          value = value.replace(spChar, '');
          return value;
      }    
      return value;
  }    
     
  
  /**
   * 포커스 이동
   */
  function keyPadUp(id) {
      var keyInput = $("#" + id);
      var containerH = $('#container').height();
      //alert((containerH-80) + ":" + (keyInput.offset().top) +":"+ $(window).height());
      if ((containerH-88) < keyInput.offset().top ) { //스크롤이 생기지 않는 페이지에서 가려짐 방지
          $('html, body').stop().animate({
              scrollTop: keyInput.offset().top
          }, 800);
      }      
  }
  
  /**
   * 혜택 바우처 상세 tab 이름변경
   */
  function comChgTabName(vochId, chgName){
      $('.tab_default a').each(function(index){
          var tagObj = $('.tab_default a').eq(index);
          var tabId = tagObj.attr('tabid');
          if (tabId == vochId){
              tagObj.text(chgName);
          }
      })
  }
  
  //안드로이드WebView 에서 location.replace 동작하지 않는 문제로 우회
  function locationReplace(url) {
      if (history.replaceState) {
          history.replaceState(null, document.title, url);
          history.go(0);
      } else {
          locaton.replace(url);
      }
  }
  
  
  
