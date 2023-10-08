var actionKeyboardFocus = false;


// Tooltip title="" 상태 안내 추가 - Q10474
function ttCont() {
    var ttw = $(".wrap_tooltip");
    ttw.each(function(){
        var tta = $(".btn_tooltip01", this);
        var ttb = $("span", tta).text();
        tta.attr("title", ttb + " 툴팁 열기");
        tta.on("mouseenter focus focusin",function(){
            $(this).attr("title", ttb + " 툴팁 닫기");
        }) .on("mouseleave blur focusout", function(){
            $(this).attr("title", ttb + " 툴팁 열기");
        })
    });
    var ttw2 = $(".tooltip_wrap");
    ttw2.each(function(){
        var tta = $(".btn_tooltip01", this);
        var ttb = $("span", tta).text();
        tta.attr("title", ttb + " 툴팁 열기");
        tta.on("mouseenter focus focusin",function(){
            $(this).attr("title", ttb + " 툴팁 닫기");
        }) .on("mouseleave blur focusout", function(){
            $(this).attr("title", ttb + " 툴팁 열기");
        })
    });

}
jQuery(document).ready(function(){
    ttCont();
});


// Tab 초점이동- Q10474
if (window.NodeList && !NodeList.prototype.forEach) {
    NodeList.prototype.forEach = Array.prototype.forEach;
}
window.addEventListener('DOMContentLoaded', function () { //2022-03-28 GGCN45 : 탭리스트 오류 수정(웹접근성) tablist01Focus ~  tablist04Focus 추가
    var tabFocus = 0;
        tablist01Focus = 0,
        tablist02Focus = 0,
        tablist03Focus = 0,
        tablist04Focus = 0,
        tabs = document.querySelectorAll('[data-role="tab"]'),
        tabs01 = document.querySelectorAll('[data-role="tab01"]'),
        tabs02 = document.querySelectorAll('[data-role="tab02"]'),
        tabs03 = document.querySelectorAll('[data-role="tab03"]'),
        tabs04 = document.querySelectorAll('[data-role="tab04"]'),
        tabList = document.querySelector('[data-role="tablist"]'),
        tabList01 = document.querySelector('[data-role="tablist01"]'),
        tabList02 = document.querySelector('[data-role="tablist02"]'),
        tabList03 = document.querySelector('[data-role="tablist03"]'),
        tabList04 = document.querySelector('[data-role="tablist04"]');

    tabs.forEach(function (tab) {
        tab.addEventListener('click', changeTabs);
    });
    tabs01.forEach(function (tab01) {
        tab01.addEventListener('click', changeTabs);
    });
    tabs02.forEach(function (tab02) {
        tab02.addEventListener('click', changeTabs);
    });
    tabs03.forEach(function (tab03) {
        tab03.addEventListener('click', changeTabs);
    });
    tabs04.forEach(function (tab04) {
        tab04.addEventListener('click', changeTabs);
    });
    if(tabList){//2022-03-28 GGCN45 :tabs -> tabList
        tabList.addEventListener('keydown', function (e) {
            tabs = document.querySelectorAll('[data-role="tab"]'); //2022-03-29 tablist 클릭시 갱신하는 경우 작동안함 수정, 2022-04-06 data-role로 변경
            if (e.keyCode === 37 || e.keyCode === 38 || e.keyCode === 39 || e.keyCode === 40) {
                // tabs[tabFocus].setAttribute('tabindex', -1);
                if (e.keyCode === 39 || e.keyCode === 40) {
                    tabFocus++;
                    if (tabFocus >= tabs.length) {
                        tabFocus = 0;
                    }
                } else if (e.keyCode === 37 || e.keyCode === 38) {
                    tabFocus--;
                    if (tabFocus < 0) {
                        tabFocus = tabs.length - 1;
                    }
                }
                // tabs[tabFocus].setAttribute('tabindex', 0);
                tabs[tabFocus].focus();
            }
        });
    }
    if(tabList01){ //2022-03-28 GGCN45 :tab01 -> tabList01
        tabList01.addEventListener('keydown', function (e) {
                tabs01 = document.querySelectorAll('[data-role="tab01"]'); //2022-03-29 tablist 클릭시 갱신하는 경우 작동안함 수정
            if (e.keyCode === 37 || e.keyCode === 39) {
                // tabs01[tablist01Focus].setAttribute('tabindex', -1);
                if (e.keyCode === 39) {
                    tablist01Focus++;
                    if (tablist01Focus >= tabs01.length) {
                        tablist01Focus = 0;
                    }
                } else if (e.keyCode === 37) {
                    tablist01Focus--;
                    if (tablist01Focus < 0) {
                        tablist01Focus = tabs01.length - 1;
                    }
                }
                // tabs01[tablist01Focus].setAttribute('tabindex', 0);
                tabs01[tablist01Focus].focus();
            }
        });
    }
    if(tabList02){//2022-03-28 GGCN45 :tab01 -> tabList01
        tabList02.addEventListener('keydown', function (e) {
            tabs02 = document.querySelectorAll('[data-role="tab02"]');  //2022-03-29 tablist 클릭시 갱신하는 경우 작동안함 수정
            if (e.keyCode === 37 || e.keyCode === 39) {
                // tabs02[tablist02Focus].setAttribute('tabindex', -1);
                if (e.keyCode === 39) {
                    tablist02Focus++;
                    if (tablist02Focus >= tabs02.length) {
                        tablist02Focus = 0;
                    }
                } else if (e.keyCode === 37) {
                    tablist02Focus--;
                    if (tablist02Focus < 0) {
                        tablist02Focus = tabs02.length - 1;
                    }
                }
                // tabs02[tablist02Focus].setAttribute('tabindex', 0);
                tabs02[tablist02Focus].focus();
            }
        });
    }
    if(tabList03){//2022-03-28 GGCN45 :tab01 -> tabList01
        tabList03.addEventListener('keydown', function (e) {
            if (e.keyCode === 37 || e.keyCode === 39) {
                tabs03 = document.querySelectorAll('[data-role="tab03"]');   //2022-03-29 tablist 클릭시 갱신하는 경우 작동안함 수정
                // tabs03[tablist03Focus].setAttribute('tabindex', -1);
                if (e.keyCode === 39) {
                    tablist03Focus++;
                    if (tablist03Focus >= tabs03.length) {
                        tablist03Focus = 0;
                    }
                } else if (e.keyCode === 37) {
                    tablist03Focus--;
                    if (tablist03Focus < 0) {
                        tablist03Focus = tabs03.length - 1;
                    }
                }
                // tabs03[tablist03Focus].setAttribute('tabindex', 0);
                tabs03[tablist03Focus].focus();
            }
        });
    }
    if(tabList04){//2022-03-28 GGCN45 :tab01 -> tabList01
        tabList04.addEventListener('keydown', function (e) {
            tabs04 = document.querySelectorAll('[data-role="tab04"]');   //2022-03-29 tablist 클릭시 갱신하는 경우 작동안함 수정
            if (e.keyCode === 37 || e.keyCode === 39) {
                // tabs04[tablist04Focus].setAttribute('tabindex', -1);
                if (e.keyCode === 39) {
                    tablist04Focus++;
                    if (tablist04Focus >= tabs04.length) {
                        tablist04Focus = 0;
                    }
                } else if (e.keyCode === 37) {
                    tablist04Focus--;
                    if (tablist04Focus < 0) {
                        tablist04Focus = tabs04.length - 1;
                    }
                }
                // tabs04[tablist04Focus].setAttribute('tabindex', 0);
                tabs04[tablist04Focus].focus();
            }
        });
    }
});
function changeTabs(e) {
    var target = e.target;
    if(target.id === ''){//이벤트 타겟이 a가 아닐때 a로 조정
        target = target.parentNode;
    }
    var parent = target.parentNode;
    var grandparent = parent.parentNode;
    //var j = target.classList;

    /*
    parent.querySelectorAll('[aria-selected="true"]').forEach(function (t) {
        t.setAttribute('title', '선택됨');
        return t.setAttribute('aria-selected', false);
    });

    parent.querySelectorAll('[aria-selected="false"]').forEach(function (t) {
        t.removeAttribute('title');
        if(t.getAttribute('class')=='current'){
            t.classList.remove('current');
        };
        return t.setAttribute('aria-selected', true);
    });
    */
//    grandparent.querySelectorAll('[aria-selected="true"]').forEach(function (t) {
//         t.removeAttribute('title');
//         // t.setAttribute('tabindex' , '-1');
//         return t.setAttribute('aria-selected', false);
//     });

    // parent.querySelectorAll('[aria-selected="false"]').forEach(function () {
    //     target.setAttribute('title', '선택됨');
    //     // target.setAttribute('tabindex' , '0');
    //     return target.setAttribute('aria-selected', true);
    // });

    if(grandparent.querySelectorAll('style').length){
        grandparent.querySelectorAll('[data-role="tabpanel"]').forEach(function (p) {
            return p.setAttribute('style', 'display:none;');
        });
        grandparent.parentNode.querySelector('#' + target.getAttribute('aria-controls')).removeAttribute('style');
    }

}
//tab reset 함수 추가
function resetTab(idx){
    var target = $('#' + idx);
    var parent = target.closest('[data-role*="tablist"]');
    parent.find('a').each(function(){
        $(this).removeAttr('title').attr({'aria-selected': false });
    });
    target.attr({'aria-selected': true ,'title':'선택됨'});
}

// UserAgent 추가 - Q10071
function browserCheck(){
    var ua = navigator.userAgent.toLowerCase();
    var word;
    var version = "N/A";

    if (ua.indexOf('msie')>-1) {
        word = 'msie ';
        var reg = new RegExp( word + '([0-9]{1,})(\\.{0,}[0-9]{0,1})' );
        if(reg.exec( ua ) != null){
            version = parseInt(RegExp.$1 + RegExp.$2);
        }
        $('html').addClass('ie ie' + version);
    } else if(ua.indexOf('trident')>-1) {
        $('html').addClass('ie ie11');
    } else if (ua.indexOf('edge')>-1) {
        $('html').addClass('edge');
    } else if(ua.indexOf('whale') >-1) {
        $('html').addClass('whale');
    } else if(ua.indexOf('chrome') >-1) {
        $('html').addClass('chrome');
    } else if(ua.indexOf('firefox') >-1) {
        $('html').addClass('firefox');
    }
}

var focusableElements ="a[href], area[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]), iframe, object, embed, *[tabindex], *[contenteditable]";
var returnFocusPopup = [];
var returnFocusConfirm = [];
var accodYn;//동적 로딩시 아코디언 script 중복 체크를 위해 요소 추가

var commonUi = {
    init: function(){
        browserCheck();
        if(!$('.user-skip').length){this.userSkip.init();}
        if($('header').length){this.gnb.init();}//gnb
        if($('.accodWrap').length) {this.accodiran.init() }//accodiran
        if($('.scrBarWrap').length) {this.scrCustomBar.init('start') }//scrCustomBar
        if($('.box_select').length) {this.expanded_select.init() }//expanded_select
        if($('.box_input01').length) {this.inp.init(); this.deleteTxt.init();}
        if($('.modal_pop').length) {this.layerPop.init()}
        if($('.list_category01').length) {this.listCate.init()}
        if($('.list_chkrdo01').length) {this.listChkrdo.init()}
        if($('.box_table').length) {this.tableInit.init()}
        if ($('.box_sorting01').length) { this.sorting.init() }
        if ($('img.svg').length) { this.cardSvg.init() }
        if ($('.modal_card_view').length) { this.cardPop.init() }
    },
    userSkip: {
        init: function(){
            var html = '<div class="user-skip" id="userSkip">';
            html += '<a href="#pcMenu01"><span>메인 메뉴 바로가기</span></a>';
            html += '<a href="#container"><span>본문 영역 바로가기</span></a>';
            html += '<a href="#footer"><span>하단 영역 바로가기</span></a>';
            html += '</div>';
            $('body').prepend(html);
        }
    },
    gnb : { //gnb
        $header : 'header',
        header : '.header',
        headerH1: 'header > h1',
        listDep2: '.list_dep2',
        btnDep2: '.btn_dep2',
        boxDep3: '.box_dep3',
        listDep3Li: '.list_dep3 > li > a',
        boxEtc : '.box_etc',
        img_banner : '.img_banner a',
        gnbBannerInitNum : 0,
        gnbBannerFirstInit : 0,
        gnbBannerInitAutoPlay: {},
        gnbBannerLastIndex : {},
        init: function(){
            var _this = this;
            _this.event();

            setTimeout(function () {

            }, 500);
        },
        gnbHeader: function() {
            var _this = this;
            var headerH = $(_this.$header).find(_this.header).height();
            if(!$(_this.$header).find('.box_gnb_banner').length || $(_this.$header).find('.box_gnb_banner').css('display') === 'none') {
                $(_this.$header).css('min-height', headerH + $(_this.$header).find('.header_sub_in').height());
            } else {
                $(_this.$header).css('min-height', headerH + $(_this.$header).find('.box_gnb_banner').height());
            }
        },
        event : function($this){  //2022-07-07 GGCN45 : 전체 수정 : [공통]GNB 카드 메뉴 내 슬라이드 SSG ED2 배너 제작 요청의 건
            var _this = this;
            var selector = {    // Q10582: 22.02.05 셀렉터 변수 추가
                body: $('body'),
                header: $(_this.header),
                boxDep3: $(_this.boxDep3),
                btnDep2: $(_this.btnDep2),
                boxGnbBanner: $('.box_gnb_banner'),
                headerSub: $('.header_sub'),
                headerH1: $(_this.header).find("h1"),
                listEtc: $(_this.boxEtc).find('.list_etc'),
                listEtcBtn: $(_this.boxEtc).find(".list_etc a"),
                inputCellBox: $(_this.boxEtc).find(".input_cell_box")
            };
            var gnbBannerListRandom = function(id){
                var $mennuListEL = $('#'+ id),
                    $gnbBannerContainer = $mennuListEL.find('.gnb-banner ul'),
                    $gnbBannerList = $gnbBannerContainer.find('li'),
                    gnbBannerListLength = $gnbBannerContainer.find('li').length,
                    swiperWrapperEl = document.createElement('ul'),
                    arrIndex = [],
                    arrChoice = [];
                for (var i = 0; i < gnbBannerListLength; i++) {
                    arrIndex.push(i);
                }
                arrChoice = shuffle(arrIndex);
                console.log(arrChoice);
                for (var i = 0; i < arrChoice.length ; i++) {
                    swiperWrapperEl.appendChild($gnbBannerList[Number(arrChoice[i])]);
                }
                $gnbBannerContainer.html($(swiperWrapperEl).html());
            }
            var gnbBannerSwiperFn = function(id){
                var bannerOptDefault = {random : false},
                    bannerOpt = $.extend({}, bannerOptDefault, $('#'+ id +' .gnb-banner').data("bannerOpt") || {});;
                if(_this.gnbBannerFirstInit == 0 && bannerOpt.random == true){
                    gnbBannerListRandom(id);
                    _this.gnbBannerFirstInit ++
                }
                window['gnbBannerSwiper_' + id] = new Swiper('#'+ id +' .gnb-banner', {
                    on : {
                        init : function(){
                            var _thisSwiper = this;
                            if(_this.gnbBannerLastIndex[id]){
                                _thisSwiper.slideTo(_this.gnbBannerLastIndex[id]);
                            }
                            // console.log(_this.gnbBannerInitAutoPlay[id]);
                            if(_this.gnbBannerInitAutoPlay[id] === false){
                                var $autoPlayBtn =  $(_thisSwiper.el).find('.swiper-button-autoplay');
                                setTimeout(function(){
                                    _thisSwiper.autoplay.stop();
                                },300);
                                _this.gnbBannerInitAutoPlay[id] = false;
                                $autoPlayBtn.addClass('stop').find('span').text('다시재생');
                            }else{
                                _this.gnbBannerInitAutoPlay[id] = true;
                            }
                            $(_thisSwiper.el).addClass('active');
                            $(_thisSwiper.el).find('.swiper-slide-duplicate').find('a').attr('tabindex','-1')
                            $(_thisSwiper.el).find('.swiper-slide').off('mouseenter.gnbBannerSwiper').on('mouseenter.gnbBannerSwiper', function (e) {
                                _thisSwiper.autoplay.stop();
                            });
                            $(_thisSwiper.el).find('.swiper-slide').off('mouseleave.gnbBannerSwiper').on('mouseleave.gnbBannerSwiper', function (e) {
                                if (!$(_thisSwiper.el).find('.swiper-button-autoplay').hasClass('stop')) {
                                    _thisSwiper.autoplay.start();
                                    _thisSwiper.autoplay.run();
                                }
                            });
                            $(_thisSwiper.el).find('.swiper-button-autoplay').off('click.gnbBannerSwiper').on('click.gnbBannerSwiper', function () {
                                var $thisEL =  $(this);
                                if (!$thisEL.hasClass('stop')) {
                                    _thisSwiper.autoplay.stop();
                                    _this.gnbBannerInitAutoPlay[id] = false;
                                    $thisEL.addClass('stop').find('span').text('다시재생');
                                    console.log(_this.gnbBannerInitAutoPlay[id]);
                                } else {
                                    _thisSwiper.autoplay.start();
                                    _thisSwiper.autoplay.run();
                                    _this.gnbBannerInitAutoPlay[id] = true;
                                    $thisEL.removeClass('stop').find('span').text('일시정지');
                                    console.log(_this.gnbBannerInitAutoPlay[id]);
                                }
                            });
                            $(_thisSwiper.el).closest('li').on('mouseleave.gnbBannerSwiper', function(e){ // gnb leave
                                swiperDestroy();
                            });
                            $(_thisSwiper.el).find('.swiper-slide a').off('focusin.gnbBannerSwiper, focus.gnbBannerSwiper').on('focusin.gnbBannerSwiper, focus.gnbBannerSwiper',  function () { // 접근성
                                _thisSwiper.autoplay.stop();
                                _thisSwiper.slideTo(1);
                                $(_thisSwiper.el).css('overflow', 'hidden');
                                $(_thisSwiper.el).find('.swiper-slide').css('opacity', '1');
                                $(_thisSwiper.el).find('.swiper-button-autoplay').addClass('stop').find('span').text('재생하기');
                            });
                            $(_thisSwiper.el).find('.swiper-slide a').off('focusout.gnbBannerSwiper').on('focusout.gnbBannerSwiper',  function () { // 접근성
                                setTimeout(function(){
                                    if(!$(document.activeElement).parent().hasClass('swiper-slide')){
                                        $(_thisSwiper.el).removeAttr('style').scrollLeft(0);
                                    }
                                },100)
                            });
                            $(_thisSwiper.el).find('.swiper-pagination span:last-child').off('focusout.gnbBannerSwiper').on('focusout.gnbBannerSwiper', function(){ // gnb leave //2022-07-05 GGCN45 :  GNB BANNER swiper 기능에 따른 수정
                                _this.gnbBannerLastIndex[id] = _thisSwiper.activeIndex;

                                _thisSwiper.destroy();
                                _this.gnbBannerInitNum  = 0;
                                var $box_dep3 = $(this).closest('.box_dep3');
                                $box_dep3.closest('li').find('.btn_dep2').removeClass('on');
                                $box_dep3.removeClass('on').hide();
                            });
                        }
                    },
                    // speed : 1000,
                    // effect : 'fade',
                    // fadeEffect : {
                    //     crossFade : true
                    // },
                    autoplay: {
                        delay: 3000, // 2022-08-25 GGU282 - 수정 : 5000 → 3000
                        disableOnInteraction: false,
                    },
                    pagination: {
                        el: '.swiper-pagination',
                        type: 'bullets',
                        clickable: true
                    },
                    a11y: {
                        prevSlideMessage: '이전 슬라이드',
                        nextSlideMessage: '다음 슬라이드',
                        slideLabelMessage: '총 {{slidesLength}}장의 슬라이드 중 {{index}}번 슬라이드 입니다.'
                    },
                    loop: true,
                });
            };
            function swiperDestroy(){
                if($('.box_dep3.on').find('.gnb-banner').length > 0){
                    var listTarget = $('.box_dep3.on').closest('li').attr('id');
                    if(!window['gnbBannerSwiper_' + listTarget] || window['gnbBannerSwiper_' + listTarget].destroyed === true){ // 2022-08-25 GGU282 - 수정
                        return;
                    }
                    _this.gnbBannerLastIndex[listTarget] = window['gnbBannerSwiper_' + listTarget].activeIndex;
                    $(window['gnbBannerSwiper_' + listTarget].el).removeAttr('style').removeClass('swiper-container-fade').scrollLeft(0);
                    window['gnbBannerSwiper_' + listTarget].destroy();
                }
            };
            function gnbNone(){
                swiperDestroy(); //gnbBanner Swiper destroy;
                $(_this.header + ', ' + _this.btnDep2 + ', ' + _this.boxDep3).removeClass('on');
                $(_this.header).removeClass('no_dep3'); // 2023-03-16 GGU282 [공통]GNB내 크림페이 메뉴 추가의 건 - 추가
                $(_this.boxDep3).hide();
            };
            $(document).on('mouseenter focusin click', _this.btnDep2, function(e){ // gnb hover
                var $this = $(this);
                console.log(e.type);
                $(_this.header).addClass('on');
                // 2023-03-16 GGU282 [공통]GNB내 크림페이 메뉴 추가의 건 - 추가 s
                if($this.attr('id') == 'pcMenu07') {
                    $(_this.header).addClass('no_dep3');
                } else {
                    $(_this.header).removeClass('no_dep3');
                }
                // 2023-03-16 GGU282 [공통]GNB내 크림페이 메뉴 추가의 건 - 추가 e
                if(!$this.is('.on')){
                    $this.parent('li').siblings('li').find(_this.btnDep2).removeClass('on');
                    $this.addClass('on');
                    $this.parent('li').siblings('li').find(_this.boxDep3).removeClass('on').hide();
                    $this.parent('li').find(_this.boxDep3).show().addClass('on');
                    // Q10071 2022-01-07: 높이 가변으로 움직이기 위해 추가
                    var _height = $this.parent('li').find('.box_dep3_in').outerHeight();
                    $this.parent('li').find('.box_favor').css('height', _height);

                    /* 2022-08-25 GGU282 - 추가 */
                    if($this.siblings('.box_dep3').find('.gnb-banner').length > 0){ // gnbBanner Swiper Init;
                        var swiperId = $this.closest('li').attr('id');
                        if(window['gnbBannerSwiper_' + swiperId] && window['gnbBannerSwiper_' + swiperId].destroyed != true){
                            window['gnbBannerSwiper_' + swiperId].destroy();
                        }
                        setTimeout(function(){
                            gnbBannerSwiperFn(swiperId);
                        }, 10)
                    }
                    /* //2022-08-25 GGU282 - 추가 */
                }
                /* 2022-08-25 GGU282 - 삭제
                if($this.siblings('.box_dep3').find('.gnb-banner').length > 0){ // gnbBanner Swiper Init;
                    var swiperId = $this.closest('li').attr('id');
                    setTimeout(function(){
                        if(!window['gnbBannerSwiper_' + swiperId]){
                            gnbBannerSwiperFn(swiperId);
                            return;
                        }
                        if(window['gnbBannerSwiper_' + swiperId].destroyed === true){
                            gnbBannerSwiperFn(swiperId);
                        }
                    }, 10)
                } */
            });
            $(document).on('mouseleave', _this.listDep2, function(e){ // gnb leave
                gnbNone();
            });
            $(document).on('focusout', _this.img_banner, function(){ // gnb leave //2022-07-05 GGCN45 :  GNB BANNER swiper 기능에 따른 수정
                if($(this).closest('.gnb-banner').length === 0){
                    gnbNone();
                }
            });
            $(document).on('focusin', _this.boxEtc, function(){
                gnbNone();
            });
            $(window).on('scroll', function(){
                var scrollTop = $(this).scrollTop();
                // var headerH = $('body').find('header').height(); // Q10071 2021-11-24: 스크롤시 튕기는 현상으로 header 높이를 잡음
                var headerH = 80; // Q10582: 22.02.05 고정 값 추가

                // Q10582: 22.02.03 스크롤 인터렉션 스크립트 변수 추가
                var inlineVal = {
                    heightVal: headerH - (scrollTop * (headerH - 48) / headerH), // Q10582: 22.02.05 계산식 변경
                    depsVal: 18 - (scrollTop * (18 - 14) / headerH), // Q10582: 22.02.05 계산식 변경
                    logoVal: {
                        width: 138 - (scrollTop*0.23),
                        height: 28 - (scrollTop*0.04616),
                        backgroundSize: (138 - (scrollTop*0.23))+"px " + (28 - (scrollTop*0.04616))+"px"
                    },
                    listEtcVal: {
                        opacity: 1 - (scrollTop*0.01),
                        fontSize: 14 - (scrollTop*0.008),
                        height: 40 - (scrollTop*0.0616)
                    }
                };
                // Q10582: 22.02.04 연산자 > 수정
                // Q10582: 22.02.05 선택자 전체 수정, 최소 값 삭제 => 인라인 스타일 제거로 변경
                if(scrollTop >= headerH ){
                    selector.body.addClass('fixed');
                    selector.header.addClass('scr');

                    // Q10582: 22.02.03 스크롤 인터렉션 스크립트 추가
                    // selector.header.removeAttr("style");   // 2022-05-16 GGCN45 : 'scr' 클래스로만 적용
                    selector.boxDep3.removeAttr("style");
                    // if(selector.boxGnbBanner.css('display') !== 'none' && $("#topBannerWrap").length > 0) {//2022-03-25 GGCN45
                    //     // selector.boxGnbBanner.removeAttr("style");
                    //     $("#topBannerWrap").prop("style").removeProperty('top');//2022-03-25 GGCN45 : 띠배너 랜덤 등 기능 개선으로 인한 스타일 전체 제거 삭제
                    // }
                    selector.headerSub.removeAttr("style");
                    selector.btnDep2.removeAttr("style");
                    // selector.headerH1.removeAttr("style"); //  // 2022-05-16 GGCN45 : 'scr' 클래스로만 적용
                    selector.listEtc.removeAttr("style");
                    selector.listEtcBtn.removeAttr("style");
                    selector.inputCellBox.removeAttr("style");

                } else {
                    selector.body.removeClass('fixed');
                    selector.header.removeClass('scr');

                    // Q10582: 22.02.03 스크롤 인터렉션 스크립트 추가
                    // selector.header.css("height", inlineVal.heightVal)   // 2022-05-16 GGCN45 : 'scr' 클래스로만 적용
                    selector.boxDep3.css("top", inlineVal.heightVal) // Q10582: 22.02.04 추가
                    // selector.boxGnbBanner.css("top", inlineVal.heightVal)
                    selector.headerSub.css("top", inlineVal.heightVal) // Q10582: 22.02.04 추가
                    selector.btnDep2.css('fontSize', inlineVal.depsVal);
                    // selector.headerH1.css({   // 2022-05-16 GGCN45 : 'scr' 클래스로만 적용
                    //     width: inlineVal.logoVal.width,
                    //     height: inlineVal.logoVal.height,
                    //     backgroundSize: inlineVal.logoVal.backgroundSize
                    // });
                    selector.listEtc.css('opacity', inlineVal.listEtcVal.opacity);
                    selector.listEtcBtn.css('fontSize', inlineVal.listEtcVal.fontSize);
                    selector.inputCellBox.css("height", inlineVal.listEtcVal.height);
                }
            });

        }
    },
    //q10011 09-15추가 ajax로 팝업 콘텐트 로딩시에 스크롤 문제 처리
    scrbarUpdate : {
        init : function(idx , type){
            var $target = $('#' + idx);
            $target.find('.layer_body').css({ 'height' : ''})
            var chgHeight = $target.height() - 320;
            var staticHeight = $target.find('.layer_wrap').height();
            var layerHd = $target.find('.layer_head').outerHeight();
            var layerBd = $target.find('.layer_body').outerHeight();
            var layerTn = $target.find('.layer_btn').outerHeight();
            var respondHeight = layerHd + layerBd + layerTn;
            if(type === 'respond'){
                if(respondHeight > chgHeight){
                    if($target.find('.layer_body').hasClass('scrBarWrap') === true){
                        $target.find('.layer_body').css({ 'height': chgHeight - layerHd - layerTn});
                    } else {
                        $target.find('.layer_body').addClass('scrBarWrap').css({ 'height': chgHeight - layerHd - layerTn});
                        commonUi.scrCustomBar.init('layer' , idx);
                    }
                    $target.find('.layer_body').css('padding-right' , '0');
                } else {
                    // console.log(layerBd , 'out')
                    $target.find('.layer_body').css({ 'height' : ''})
                    $target.find('.layer_body').css('padding-right' , '');
                }

            } else {
                if(staticHeight > chgHeight){
                    $target.mCustomScrollbar('update');
                }
            }
        }
    },
    scrCustomBar : { // 스크롤 Q10048 스크롤 콜백 추가 210623
        target :'.scrBarWrap',
        /*
        init : function(){
            var _this = this;

            $(this.target).each(function(){
                var scrType = $(this).attr('data-type');
                var popType = $(this).attr('data-pop-type');
                _this.event($(this), $('.scrBarWrap'), scrType, popType);
            });

        },*/
        init : function(type , idx){//팝업일때 변수값 추가 Q10011 0703
            var _this = this;

            switch(type){
                case "start" ://처음 실행시
                    $(this.target).each(function(){
                        if(!$(this).is('.layer_body') && !$(this).is('.modal_pop')){//팝업일 경우 제외
                            $(this).addClass('scrAct');//target class 세팅
                            var scrType = $(this).attr('data-type');
                            _this.event($(this), $('.scrAct'), scrType);//변경된 target class 로 수행
                        }
                    });
                break;

                case "layer" :
                    var popType = $('#' + idx).attr('data-pop-type');
                    var loadType =  $('#' + idx).attr('data-load-type');
                    if(popType === 'respond' || loadType ==='ajax'){//ajax로 static 로드시 eventTarget이 repond랑 동일해짐
                        var eventTarget = $('#' + idx).find('.scrBarWrap');//respond일때 eventTarget
                        var scrType = eventTarget.attr('data-type');
                    } else {

                        var eventTarget = $('#' + idx);//static일때 eventTarget
                        var scrType = eventTarget.attr('data-type');
                    }
                    _this.event($(this) , eventTarget , scrType , popType , idx);
                break;
            }

        },
        /*
        event : function($this, $target, type, type2){
            var _this = this;
            var option;
            if(type == 'scrSel'){
                option = {
                    horizontalScroll : false,
                    setTop: 0,
                    theme : 'dark',
                    mouseWheelPixels : 500,
                    autoHideScrollbar : true,
                    advanced:{
                        updateOnContentResize: true
                    },
                    callbacks:{
                        whileScrolling: function () {

                            if (type2 == 'static') { //static 타입 팝업
                                if(parseInt($('.mCSB_container').css('top')) < -160){
                                    $('.layer_close').addClass('fixed');
                                }else {
                                    $('.layer_close').removeClass('fixed');
                                }
                            }
                            if ($(this).hasClass('layer_body')) { //respond 타입 팝업
                                if (parseInt($('.mCSB_container').css('top')) < 0) {
                                    $('.layer_head').addClass('fixed');
                                } else {
                                    $('.layer_head').removeClass('fixed');
                                }
                            }
                        }
                    },
                }
            }*/
        event : function($this, $target, type, type2 , prtId){//210705 Q10011 정적 팝업 해당 id로 동작하게 수정 210705
            var _this = this;
            var option = {
                    horizontalScroll : false,
                    setTop: 0,
                    theme : 'minimal-dark',
                    autoHideScrollbar : true,
                    advanced:{
                        updateOnContentResize: true
                    },
                    mouseWheel: { //스크롤 끝 또는 시작에 도달하면 부모요소를 자동으로 스크롤하는것 방지
                        preventDefault:true
                    },
                    mouseWheelPixels : 200,
                    callbacks:{
                        whileScrolling: function () {
                            // 2023-01-06 GGU442 (카드상세)UI Audit__상세 팝업 내 BG_SCROLL시 X버튼 이동되는 현상 수정
                            // if (type2 == 'static') { //static 타입 팝업
                            //     if(parseInt($('#'+prtId + ' .mCSB_container').css('top')) < -160){
                            //         $('#' + prtId + ' .layer_close').addClass('fixed');
                            //     }else {
                            //         $('#' + prtId + ' .layer_close').removeClass('fixed');
                            //     }
                            // }
                            
                            //if ($(this).hasClass('layer_body')) { //respond 타입 팝업
                            if (type2 == 'respond') {// q10011 타입 수정
                                if (parseInt($('#' + prtId + ' .mCSB_container').css('top')) < 0) {
                                    $('#' + prtId + ' .layer_head').addClass('fixed');

                                } else {
                                    $('#' + prtId + ' .layer_head').removeClass('fixed');
                                }
                            }

                            if ($(_this.target).hasClass('swiper-slide')){
                                if (parseInt($('.swiper-slide-active .mCSB_container').css('top')) < 0) {
                                    $(this).parents('.layer_wrap').find('.layer_head').addClass('fixed');
                                } else {
                                    $(this).parents('.layer_wrap').find('.layer_head').removeClass('fixed');
                                }
                            }
                        }
                    },
                }

                $target.mCustomScrollbar(option);

        }
    },
    expanded_select : { //select
        /* 2022-05-06 GGU282 [혜택]클럽서비스_클럽호텔 내 드롭다운형 다중체크박스 UI 추가 건 - 관련 내용 수정 및 추가
            - 다중체크박스 클래스 - .box_select.multi_select
            - 다중체크박스 적용 버튼 - .btn_set - 다중 선택 시 적용 버튼을 눌러야 적용됨
            - 다중체크박스 편집 중 - .box_select.multi_editing
            - 다중체크박스 편집 중일 때 기존 선택 항목 - input.pre_checked
        */
        expandedWrap : '.box_select',
        expandedAnchor : '.box_select .drop_link a',
        select_down : '.box_select .select_down a',
        expandedValue : '.box_select .select_down a',
        expandedBtnSet : '.box_select .select_down .btn_set', // 2022-05-06 GGU282 - 적용 버튼 추가
        init : function(){
            var _this = this;
            $(this.expandedWrap).removeClass('on');
            _this.event();

            $(".box_select .select_down").each(function(){
                if($(this).find(".selected").length > 0){
                    var $selected = $(this).find(".selected");
                    var $selectedhtml = $selected.find("> a").html();
                    $selected.closest(".box_select").find(".drop_link a").html($selectedhtml);
                }
                $(this).find("li.active").find('a').attr('title','선택됨'); // 2023-01-27 GGU282 - 웹접근성 추가. a title 처리 추가
            });
        },
        event : function(e){
            var _this = this;
            $(document).on('click', _this.expandedAnchor, function(e){ //
                _this.action($(this).parents('.box_select'));
            });

            $(document).on('click', _this.expandedValue, function(){
                /* 2022-05-06 GGU282 - 수정 */
                /* 기존 코드
                _this.value($(this));
                */
                if(!$(this).closest('.box_select').hasClass('multi_select')) {
                    _this.value($(this));
                }
                /* //2022-05-06 GGU282 - 수정 */
            });

            /* 2022-05-06 GGU282 - 추가 */
            $(document).on('click', _this.expandedBtnSet, function(){
                if($(this).closest('.box_select').hasClass('multi_select')) {
                    _this.multi_value_set($(this));
                }
            });
            /* //2022-05-06 GGU282 - 추가 */

            var container = $(_this.expandedWrap);
            $('body').on('click',function(e){
                /*if($(e.target).closest('.box_select').length === 0 && $('.box_select').hasClass('on')){
                    _this.close()
                }*/
                //console.log(container)
                /* 2022-05-06 GGU282 - 수정 */
                /* 기존 코드
                if(!container.is(e.target) && container.has(e.target).length === 0 && container.hasClass("on")){
                    _this.close();
                }
                */
                if($(e.target).closest('.box_select.on').length == 0){
                    _this.close();
                }
                /* //2022-05-06 GGU282 - 수정 */
            });
        },
        action : function($this){
            $(this.expandedWrap).not($this).removeClass('on');
            if(!$this.hasClass('on')){
                $this.addClass('on').attr('aria-expanded','true');
                $this.find('.drop_link').attr('aria-expanded','true');
                $this.find('.select_down').slideDown(300);
                /* 2022-05-06 GGU282 - 추가 */
                if($this.hasClass('multi_select')) {
                    $this.addClass('multi_editing');
                    $this.find('.select_down li input:checked').addClass('pre_checked');
                }
                /* //2022-05-06 GGU282 - 추가 */
            } else {
                $this.removeClass('on').attr('aria-expanded','false');
                $this.find('.drop_link').attr('aria-expanded','false');
                $this.find('.select_down').slideUp(300);
                /* 2022-05-06 GGU282 - 추가 */
                if($this.hasClass('multi_select')) {
                    $this.find('.select_down li input').prop('checked', false);
                    $this.find('.select_down li input.pre_checked').removeClass('pre_checked').prop('checked', true);
                    $this.removeClass('multi_editing');
                }
                /* //2022-05-06 GGU282 - 추가 */
            }
        },
        value : function($item){
            var _this = this;
            var $val = $item.html();
            $item.closest('.box_select').find('.drop_link > a').html($val);
            $item.parents('.drop_link').attr('aria-expanded','false');
            $item.parents('.select_down').slideUp(300);
            $item.closest('.box_select').find('li').removeClass('active').find('a').attr('title',''); // 2023-01-27 GGU282 - 웹접근성 수정. a title 처리 추가
            $item.parent('li').addClass('active');
            $item.attr('title','선택됨'); // 2023-01-27 GGU282 - 웹접근성 추가. a title 처리 추가
            _this.close();
        },
        /* 2022-05-06 GGU282 - 추가 */
        multi_value_set : function($btnSet){
            var _this = this;
            var $this_wrap = $btnSet.closest('.box_select');
            var val_str = '';
            var $active_list = $this_wrap.find('.select_down li input:checked');

            $this_wrap.find('.select_down li input').removeClass('pre_checked');
            $this_wrap.removeClass('multi_editing');

            $active_list.each(function(index){
                if(index > 0) {
                    val_str += ' / ' + $(this).next('label').text();
                } else {
                    val_str = $(this).next('label').text();
                }
            });
            if($active_list.length == 0) {
                val_str = $this_wrap.find('.drop_link > a').data('title');
            }

            $this_wrap.find('.drop_link > a > span').html(val_str);
            _this.close();
        },
        /* //2022-05-06 GGU282 - 추가 */
        close : function($this){
            var _this = this;

            /* 2022-05-06 GGU282 - 추가 */
            if($('.box_select.multi_editing').length > 0) {
                $('.box_select.multi_editing').find('.select_down li input').prop('checked', false);
                $('.box_select.multi_editing').find('.select_down li input.pre_checked').removeClass('pre_checked').prop('checked', true);
                $('.box_select.multi_editing').removeClass('multi_editing');
            }
            /* //2022-05-06 GGU282 - 추가 */

            $('.box_select').removeClass('on').attr('aria-expanded','false');
            $('.drop_link').attr('aria-expanded','false'); // 2022-05-06 GGU282 - 추가
            $('.select_down').slideUp(300);
            return false;
        }
    },
    accodiran : {
        accodWrap : '.accodWrap',
        accodBtn : '.accodBtn',
        accodSlide : '.accodSlide',
        /*
        init : function(){
            var _this = this;
            _this.title();
            _this.event();
        },
        */
        init : function(){
            var _this = this;
            if(accodYn !== true){
                _this.title();
                _this.event();
                // console.log('아코디언 최초 실행');
                accodYn = true;
            } else {
                // console.log('아코디언 중복 실행');
                //return false;
            }
        },
        title : function(){
            var _this = this;
            $(_this.accodBtn).each(function(){
                var linkAttrTxt = $(this).text();
                var linkAttr =  $(this).attr('data-title');
                if($(this).closest(_this.accodWrap).is('.on') &&  $(this).closest(_this.accodWrap).find(_this.accodSlide).is(':visible')){
                    $(this).attr('title', linkAttr + '닫기');//wai: 상황에 맞는 정보 제공
                }else{
                    $(this).attr('title', linkAttr + '열기');
                }
            });
        },
        event : function(){
            var _this = this;
            $(document).on('click', _this.accodBtn, function(e){ //
                e.preventDefault();
                _this.slideUpDown($(this));
            })
        },
        slideUpDown : function($this){
            var _this = this;
            var $btn = $(_this.accodBtn);
            var linkAttrTxt = $this.text();
            var linkAttr = $this.attr('data-title');

            //Q10048 아코디언 내에 아코디언이 있는 경우 부모아코디언 열면 상속되어 자식 아코디언도 열려 수정  find-> children
            if ($this.closest(_this.accodWrap).is('.on') && $this.closest(_this.accodWrap).children(_this.accodSlide).is(':visible')){
                $this.attr('title', linkAttr + '열기');
                $this.closest(_this.accodWrap).removeClass('on').children(_this.accodSlide).slideUp('300');
            }else{
                $this.attr('title', linkAttr + '닫기');
                //$this.closest(_this.accodWrap).addClass('on').children(_this.accodSlide).slideDown('300');
                $this.closest(_this.accodWrap).addClass('on').children(_this.accodSlide).slideDown('300' , function(){
                    if($this.closest('.modal_pop').hasClass('static')){
                        commonUi.scrbarUpdate.init($this.closest('.modal_pop').attr('id') ,'static');//Q10011 0915 가변 높이에 대한 스크롤바 업데이트(static 적용);
                    } else {
                        commonUi.scrbarUpdate.init($this.closest('.modal_pop').attr('id') ,'respond'); //Q10011 0915 가변 높이에 대한 스크롤바 업데이트(static 적용);
                    }
                });
            }

            //Q10048 이승현(타이호인스트)님 요청으로 콜백함수 추가 2021-08-13
            if (typeof callback_slideUpDown === 'function') {
                callback_slideUpDown($this);
            }

        }
    },
    inp : {
        boxInp : '.box_input01',
        boxInpCellBox : '.input_cell_box',
        boxInpCell : '.input_cell',
        boxInpTxt : '.input_cell input',
        btnDelInput : '.input_cell_box .btn_del',
        init : function(){
            var _this = this;
            _this.event();
        },
        event : function(){
            var _this = this;

            $(document).on('focusin', _this.boxInpTxt, function(){
                _this.focusin($(this));
            });

            $(document).on('focusout', _this.boxInpTxt, function(){
                _this.focusout($(this));
            });

            $(document).on('keyup paste', _this.boxInpTxt, function(){
                _this.keyup($(this));
            });

            $(document).on('mousedown', _this.btnDelInput, function(){
                _this.deleteAct($(this));
            });

            $(document).on('keyup', _this.boxInpTxt, function(){
                var  $keyThis = $(this);
                if($keyThis.val().length > 0 || !$keyThis.val() == ''){
                    $keyThis.closest(_this.boxInpCellBox).addClass('on');
                }else{
                    $keyThis.closest(_this.boxInpCellBox).removeClass('on');
                }
            });

        },
        focusin : function($this){
            var _this = this;
            $this.closest(_this.boxInpCellBox).addClass('focused');
            $this.closest(_this.boxInpCellBox).removeClass('completed');
            if ($this.val().length > 0) {
                $this.closest(_this.boxInpCellBox).find(".btn_del").css("display","inline-block");
            }
        },
        focusout : function($this){
            var _this = this;
            var _notEmptyLength = 0;
            $this.closest(_this.boxInpCellBox).find("input[type='text'], input[type='password'], input[type='number'], input[type='tel']").each(function(){
                if($(this).val() !== ''){
                    _notEmptyLength = _notEmptyLength + 1;
                }
            });
            $this.closest(_this.boxInpCellBox).find(".btn_del").css("display", ""); //focusout시 무조건 삭제버튼 hide
            $this.closest(_this.boxInpCellBox).removeClass('focused');//focusout시 무조건 focuese 클래스 삭제


            if (_notEmptyLength == 0) {
                $this.closest(_this.boxInpCellBox).removeClass('focused');
                $this.closest(_this.boxInpCellBox).removeClass('completed');
            } else {
                $this.closest(_this.boxInpCellBox).addClass('completed');
            }

            /* 2021-06-08 row 안에 필드 복수일 경우가 있어 수정
            if($this.val() == ''){
                $this.closest(_this.boxInpCellBox).removeClass('focused');
            }else {
                $this.closest(_this.boxInpCellBox).addClass('focused');
            }
            */
        },
        keyup : function($this){
            var _this = this;
            var _notEmptyLength = 0;
            $this.closest(_this.boxInpCellBox).find("input[type='text'], input[type='password'], input[type='number'], input[type='tel']").each(function () {
                //Q10048 211221 tel 추가
                if($(this).val() !== ''){
                    _notEmptyLength = _notEmptyLength + 1;
                }
            });
            if(_notEmptyLength == 0){
                $this.closest(_this.boxInpCellBox).find(".btn_del").css("display","");
            }else{
                $this.closest(_this.boxInpCellBox).find(".btn_del").css("display","inline-block");
            }
            /* 2021-06-08 row 안에 필드 복수일 경우가 있어 수정
            if($this.val() == ''){
                $this.closest(_this.boxInpCellBox).find(".btn_del").css("display","");
            }else {
                $this.closest(_this.boxInpCellBox).find(".btn_del").css("display","inline-block");
            }
            */
        },
        completedAct : function($this){
            var _this = this;
            $this.closest(_this.boxInpCellBox).addClass('completed');
        },
        errorAct : function($this){
            var _this = this;
            $this.closest(_this.boxInpCellBox).addClass('error');
        },
        deleteAct : function($this){
            var _this = this;
            $this.closest(_this.boxInpCellBox).find('input[type="text"], input[type="password"], input[type="number"], input[type="tel"]').val("");
            $this.closest(_this.boxInpCellBox).find('input')[0].focus();
            $this.closest(_this.boxInpCellBox).removeClass("on completed");
            $this.css("display","");
        }
    },
    deleteTxt: { //웹접근성 이슈로 내용삭제 추가 Q10071 2022.03.04
        init:function(){
            var boxInp = $('.box_input01');
            var boxInpCellBox = boxInp.find('.input_cell_box');
            if(boxInpCellBox.length){
                boxInpCellBox.each(function(){
                    var _this = $(this);
                        boxInpCell = _this.find('.input_cell'),
                        btnDelInput = _this.find('.btn_del'),
                        delText = btnDelInput.find('span').text();

                    if (!btnDelInput.attr('role')) {
                        btnDelInput.attr('role', 'button');
                    }
                    if(delText === '내용 삭제하기' && boxInpCell.find('.input_label').length) {
                        btnDelInput.find('span').text(boxInpCell.find('.input_label').text() + ' 삭제하기');
                    }

                });
            }
        }
    },
    layerPop : {
        boxLayer : '.modal_pop',
        layerWrap : '.layer_wrap',
        layerHead : '.layer_head',
        layerBody: '.layer_body',
        layerBtn: '.layer_btn',
        layerOpen : '.layer_open',
        layerClose: '.layer_close a',
        layerFocus: '.layer_open.focus',
        init : function(){
            var _this = this;//2022-07-13 GGCN45 : 전역변수 오염 수정
            _this.event();
        },
        event : function($targetId){

            /*
            $(document).on('click', _this.layerOpen, function(e){
                e.preventDefault();
                var $targetId = $(this).attr('data-id'); // 버튼에 data-id 와 layer id 값 연결

                _this.openLayer($targetId);
                $(this).addClass('focus');
            });
            */

            /*
            $(document).on('click', _this.layerClose, function(e){
                e.preventDefault();
                var btnObj = $(this);
                var $targetId = $(this).closest(_this.boxLayer).attr('id'); // 버튼에 data-id 와 layer id 값 연결

                _this.closeLayer($targetId);
            });
            */

        },
        getScrollbarWidth : function() {
            var inner = document.createElement('p');
            inner.style.width = "100%";
            inner.style.height = "200px";

            var outer = document.createElement('div');
            outer.style.position = "absolute";
            outer.style.top = "0px";
            outer.style.left = "0px";
            outer.style.visibility = "hidden";
            outer.style.width = "200px";
            outer.style.height = "150px";
            outer.style.overflow = "hidden";
            outer.appendChild (inner);

            document.body.appendChild (outer);
            var w1 = inner.offsetWidth;
            outer.style.overflow = 'scroll';
            var w2 = inner.offsetWidth;
            if (w1 == w2) w2 = outer.clientWidth;

            document.body.removeChild(outer);

            return (w1 - w2);
        },
        openLayer : function($targetId, btnid, popAnchor , customurl){
            var $targetIdMmodal = $('#'+$targetId);
            var $body = $('body');
            var popLength = $('.modal_pop.active').length;
            $targetIdMmodal.attr('data-focus-btn' , btnid);//btnid 세팅
            $('.modal_pop .layer_close a').attr('title', '레이어 닫기');// 2023-01-26 GGU442 2023접근성 추가
            // console.log(popLength);
            if($targetIdMmodal.find('.radio_box').size()){
                $(document).on('click' , '.respond .radio_box label' , function(){
                    setTimeout(function(){commonUi.scrbarUpdate.init($targetId , 'respond')} , 500)
                });
            }
            if(popLength >= 1){
                $('.modal_pop.active').addClass('multy');
            }
            if (!$('html').hasClass('layer_active')) {
                $('body, html').addClass('layer_active');
            }
            $body.css('padding-right', commonUi.layerPop.getScrollbarWidth() + 'px');
            modalLoadType = $targetIdMmodal.attr('data-load-type');
            modalType = $targetIdMmodal.attr('data-pop-type');

            if(modalLoadType ==='ajax'){
                commonUi.layerPop.openCase($targetId, customurl);
            } else {
                $targetIdMmodal.addClass("active").attr({'aria-modal': true, 'role': 'dialog', 'tabIndex': '-1'}).removeAttr('aria-hidden').animate({'opacity':'1'}, 100,function(){
                    commonUi.layerPop.openCase($targetId, customurl);
                    $targetIdMmodal.find(commonUi.layerPop.layerWrap).attr('tabindex', 0).focus();
                });
                // static 팝업 스크롤생기면 위치값 재조정으로 딜레이
                if ($targetIdMmodal.hasClass('static', 'scrBarWrap') || $targetIdMmodal.hasClass('static', 'modal_card_view')) {
                    setTimeout(function () {
                        $targetIdMmodal.find('.modal_container').css('opacity','1');
                    }, 300);
                }
            }

            if (popAnchor) {

                if ($targetIdMmodal.hasClass('modal_card_view')) {// 카드상세 팝업만 적용
                    if ($targetIdMmodal.find(".tab_default").length == 1) { //tab 팝업인 경우

                        $(popAnchor).click(); //해당 탭 클릭

                        var tabCon = $(popAnchor).attr('href');

                        if ($(tabCon).find(".accodWrap").length > 1) { //active 된 탭 컨텐츠 내에 아코디언 있는 경우 첫번째 아코디언 열림

                            if ($(tabCon).find(".accodWrap").hasClass('on')) { //아코디언 열려있으면 모두 닫고 첫번째만 열기
                                $(tabCon).find(".accodWrap").removeClass('on').children('.accodSlide').hide();
                                $(tabCon).find(".accodWrap:first-child a").click();
                            } else {
                                $(tabCon).find(".accodWrap").removeClass('on');
                                $(tabCon).find(".accodWrap:first-child").hasClass('on');
                                $(tabCon).find(".accodWrap:first-child a").click();
                            }

                        } else {
                            $targetIdMmodal.find(".scrBarWrap").mCustomScrollbar('scrollTo', 0); //아코디언 없으면 스크롤 상단으로 이동
                        }

                    } else {
                        //$(popAnchor).click();
                        /* Q10293 팝업 오픈시 아코디언 호출이 겹치는 부분 수정 */
                        if ($(popAnchor).hasClass('card_bundle') && !$(popAnchor).closest('accodWrap').hasClass('on') || $(popAnchor).closest('.accodWrap').siblings('.accodWrap').hasClass('on')) { // Q10293 해당 영역 호출시 다른 아코디언 열려있는 부분 수정
                            $(popAnchor).closest('.accodWrap').addClass('on').children('.accodSlide').slideDown('300');
                            $(popAnchor).closest('.accodWrap').siblings('.accodWrap').removeClass('on').children('.accodSlide').hide();
                        }
                        else if (!$(popAnchor).closest('.accodWrap').hasClass('on'))
                            $(popAnchor).click();
                        setTimeout(function () {
                            $targetIdMmodal.find(".scrBarWrap").mCustomScrollbar('scrollTo', popAnchor);
                        }, 500);
                    }
                }



                if ($targetIdMmodal.hasClass("modal_product")) { // 금융 상품 상세 팝업인 경우
                    if (!$(popAnchor).closest('accodWrap').hasClass('on') || $(popAnchor).closest('.accodWrap').siblings('.accodWrap').hasClass('on') || !$targetIdMmodal.find(".tab_default").children('a').ep(0).hasClass('current')) {
                        // 아코디언초기화
                        $(popAnchor).closest('.accodWrap').addClass('on').children('.accodSlide').slideDown('300');
                        $(popAnchor).closest('.accodWrap').siblings('.accodWrap').removeClass('on').children('.accodSlide').hide();
                        //자주찾는 질문 초기화
                        $targetIdMmodal.find('.qna_list > .accodWrap').removeClass('on').children('.accodSlide').hide();
                        //tab 초기화
                        $targetIdMmodal.find(".tab_default").children('a').eq(0).addClass('current').siblings('a').removeClass('current');
                        $targetIdMmodal.find(".tab_content").eq(0).show().siblings().hide();
                    } else if (!$(popAnchor).closest('.accodWrap').hasClass('on')) {
                        $(popAnchor).click();
                    }

                }

            }

            if($targetIdMmodal.hasClass('scrBarWrap', 'mCustomScrollbar')) { // Q10071 - 추가
                $targetIdMmodal.mCustomScrollbar('destroy'); // 초기화
            }

            /* q10011 멀티 팝업 오픈시 개별 클로우징 작업 및 dim 하나만 적용
                Q10071 드래그시에 dim영역 놓았을 때 닫히는 현상 이슈로
                이벤트 click => mousedown으로 변경
            */
            $(document).on('mousedown' , '.modal_pop:not(.modal_alert):not(.not_dim)' , function(e){
                var myTarget = $(e.target);
                if($(myTarget).parents('#container').hasClass('standard')){ // 2022-01-10 표준화일때 배경 클릭 x 예외처리 Q10011
                    return;
                }
                if(myTarget.hasClass('active')){
                    $('.modal_pop').css({'opacity':'0'}).removeClass('active multy').attr('aria-hidden', true).find(commonUi.layerPop.layerWrap).attr('tabindex', -1).end().find('.layer_close').removeClass('fixed'); //2022-01-09 Q10071: aria-hidden 속성 추가
                    $('body').removeAttr('style');
                    $('body, html').removeClass('layer_active');
                }

            });
            /*/ Q10071 backdrop 클릭시 닫기 추가
            $(document).on('mouseup', '.modal_pop:not(.modal_alert).active' ,function(e){ //알럿 팝업 제외
                if($targetIdMmodal.find('.modal_container').has(e.target).length === 0) {
                    $targetIdMmodal.css({'opacity':'0'}).removeClass('active').removeAttr('tabindex').find('.layer_close').removeClass('fixed');
                    $body.removeAttr('style');
                    $('body, html').removeClass('layer_active');

                    if($targetIdMmodal.find('.nppfs-keypad').length) { //가상 키패드
                        $targetIdMmodal.find('.nppfs-keypad').css('display', 'none');
                    }
                }
            });
            */
            // 2023-01-25 Q11067 #274 웹접근성 (신분증인증-전체동의팝업-5번약관 광고성매체 체크포커스시 시각적표시)
            if ($targetIdMmodal.find(".pop_in_box").length > 0){
                $(".scrBarWrap .pop_in_box input[type=checkbox]").on("focus",function(){
                    var _this = $(this)
                        , _scrollTarget = _this.closest(".scrBarWrap")
                        , _boxTop = $(this).parents(".pop_in_box").position().top
                        ;
                    //console.log(_boxTop);
                    _scrollTarget.mCustomScrollbar('scrollTo', _boxTop);
                });
            }
        },
        /*
        closeLayer : function($targetId, focusid){
            var _this = this;
            var $btnObj = $("#" + focusid);
            var $targetIdMmodal = $('#'+$targetId);

            $('body, html').removeClass('layer_active');
            $('body').removeAttr('style');
            $targetIdMmodal.css({'opacity':'0'}).removeClass('active').removeAttr('tabindex');
            $(_this.layerBody).removeClass('scrBarWrap').css({'height' : ''});
            $(_this.boxLayer).find('.modal_wrap').css({'padding' : ''});
            if(focusid){
                $btnObj.focus();
            }

        },
        */
        closeLayer : function($targetId, focusid){
            var _this = this;
            var $btnObj = $("#" + focusid);
            var $targetIdMmodal = $('#'+$targetId);
            if(focusid === undefined || focusid == ''){
				focusid = $targetIdMmodal.attr('data-focus-btn');//focusid를 공백이나 넣지 않았을때 값 변경
			}
			var $focusBtn = $('[data-popup="true"]');
            $('body, html').removeClass('layer_active');
            $('body').removeAttr('style');
            $('.layer_close').removeClass('fixed');
            $targetIdMmodal.css({'opacity':'0'}).removeClass('active').removeAttr('aria-modal', 'role').attr('aria-hidden', true).hide();//접근성 이슈로 hide 추가 2022-02-23 Q10011
            $targetIdMmodal.find(commonUi.layerPop.layerWrap).removeAttr('tabindex'); //2022-01-09 Q10071: aria-hidden 속성 추가
            var popLength = $('.modal_pop.active').length - 1;
            $('.modal_pop.active').eq(popLength).removeClass('multy');

            // static 팝업 스크롤생기면 위치값 재조정으로 딜레이
            if ($targetIdMmodal.hasClass('static scrBarWrap')) {
                $targetIdMmodal.find('.modal_container').css('opacity', '0');
            }
            $(_this.boxLayer).find('.modal_wrap').css({'padding' : ''});
            if(focusid){
                $btnObj.focus();
            }
			if($btnObj.length === 0){
				$focusBtn.focus();
				$('a , button').removeAttr('data-popup');
			}
            modalLoadType = $targetIdMmodal.attr('data-load-type');
            modalType = $targetIdMmodal.attr('data-pop-type');
            if(modalLoadType ==='ajax'){
                $targetIdMmodal.empty();
            }
            if($targetIdMmodal.hasClass('scrBarWrap', 'mCustomScrollbar')) { // Q10071 - 추가
                $targetIdMmodal.mCustomScrollbar('destroy'); // 초기화
            }
            $targetIdMmodal.find('.scrBarWrap').mCustomScrollbar('scrollTo', 'top');//다중 스테틱의 경우 close 버튼이 겹쳐지는 현상이 있어 추가
        },

        openCase : function($targetId , customurl){
            var $targetIdMmodal = $('#'+$targetId);
            var chgHeight = $targetIdMmodal.find('.box_content').innerHeight() + $targetIdMmodal.find('.layer_btn').innerHeight() + $targetIdMmodal.find('.layer_head').innerHeight() + 40;
            var modalHeight = $targetIdMmodal.find('.layer_wrap').innerHeight()
            var layerHeadH = $targetIdMmodal.find('.layer_head').outerHeight();
            var layerBtnH = $targetIdMmodal.find('.layer_btn').outerHeight();
            var windowHei = $(window).innerHeight();
            chgHei = chgHeight - (windowHei - 320);

            if(modalType == 'respond'){
                //버튼이 유무시 padding 조정
                if($targetIdMmodal.find('.layer_btn').length){
                    $targetIdMmodal.find('.layer_body').css('padding-bottom', '0');
                    if ($targetIdMmodal.find('.swiper-slide').length) { //슬라이드 있을때 여백 추가
                        $targetIdMmodal.find('.swiper-slide').css('padding-bottom', '40px');
                    } else {
                        $targetIdMmodal.find('.box_content').css('padding-bottom', '40px');
                    }
                } else {
                    $targetIdMmodal.find('.layer_body').css('padding-bottom','40px');
                }

               // console.log($targetIdMmodal.find('.box_content').innerHeight(), $targetIdMmodal.find('.layer_btn').innerHeight(), $targetIdMmodal.find('.layer_head').innerHeight());
                if(chgHei >= 0){//콘텐트 내용이 클때 스크롤 적용
                    //타겟을 layer body 에서 scrBarWrap로 수정
                    $targetIdMmodal.find('.scrBarWrap').css({ 'height': modalHeight - layerHeadH - layerBtnH});
                    $targetIdMmodal.find('.scrBarWrap').css('padding-right' , '0'); //2023-04-11 Q11067 316_GPCC_팝업 스크롤 수정요청의건 (mCustom 미호출시 .layer_body 선택자 오류수정)
                } else {
                    $('#'+$targetId + '>.modal_wrap>.modal_container>.layer_wrap>.layer_body').removeClass('scrBarWrap');
                }
                //swiper타입 추가 Q10011
                if($($targetIdMmodal.find('.swiper-container:not(.c_detail)')).length !== 0){//q10011 swiper타입이 아닌경우에도 lenth 체크를 해서 일반 슬라이드와 스와이프 형태가 같이 있을경우 에러 수정
                    // 2021-11-22 초기화 세팅 개발팀 요청
                    if ($targetIdMmodal.find('.swiper-container')[0].swiper) {
                        var swiper = $targetIdMmodal.find('.swiper-container')[0].swiper;
                        if (!swiper.isBeginning) {
                            swiper.slideTo(0);
                        }
                    } else {
                        var agreeSwiper = new Swiper('#' + $targetId + ' .swiper-container', {//q10011 0721 복수개의 스와이퍼 팝업이 있는 케이스로 해당 아이디로 실행
                            loop: false,
                            navigation : {
                                prevEl : '#' + $targetId + ' .swiper-button-prev',
                                nextEl : '#' + $targetId + ' .swiper-button-next'
                            },
                            on : {
                                init: function() {
                                    var pageAll = $('#' + $targetId + ' .swiper-slide:visible').length;
                                    $('#' + $targetId + ' .page_all').text(pageAll);
                                },
                                activeIndexChange : function(){
                                    $(".scrBarWrap").mCustomScrollbar('scrollTo', '0');
                                    $('#' + $targetId + ' .page_cur').text(this.realIndex + 1);
                                   var scrTop = $targetIdMmodal.find('.swiper-slide').eq(this.realIndex).find('.mCSB_container').css('top'); //2022-07-06 GGCN45 : $('.swiper-slide') --> id.find로 수정
                                   if(parseInt(scrTop) < 0){
                                       $('#' + $targetId + ' .layer_head').addClass('fixed')
                                   } else {
                                       $('#' + $targetId + ' .layer_head').removeClass('fixed')
                                   }
                                }
                            }
                        });
                    }

                    $targetIdMmodal.find('.swiper-container').css({ 'height': windowHei - layerHeadH - layerBtnH - 320});//2022-07-06 GGCN45 : $('.swiper-slide') --> id.find로 수정
                }
            }
            commonUi.scrCustomBar.init('layer' , $targetId);

        }
    },
    listCate : {
        listCate01 : '.list_category01',
        listCateLink01 : '.list_category01 > li > a',
        init : function(){
            var _this = this;
            _this.event();
        },
        event : function(){
             var _this = this;
            $(document).on('click', _this.listCateLink01, function(e){
                e.preventDefault();
                $(this).closest(_this.listCate01).find('li').removeClass('on').find('a').removeAttr('aria-current title');
                $(this).attr({'title':'선택됨', 'aria-current':'location'}).parent('li').addClass('on');
            });
        }
    },
    listChkrdo : {
        listChkrdo01 : '.list_chkrdo01',
        listChkrdoLink01 : '.list_chkrdo01 > li label',
        listChkrdoLink02 : '.list_chkrdo01 > li input',
        init : function(){
            var _this = this;
            _this.event();
        },
        event : function(){
             var _this = this;
            $(document).on('click', _this.listChkrdoLink01, function(){
                var dataInput = $(this).prev(_this.listChkrdoLink02).attr('type');
                if(dataInput == 'radio'){
                    $(this).closest(_this.listChkrdo01).find('input').attr('checked','');
                    $(this).attr('checked','checked');
                }else if(dataInput == 'checkbox'){
                    if($(this).is(':checked')){
                        $(this).attr('checked','');
                    }else{
                        $(this).attr('checked','checked');
                    }
                }
            });
        }
    },
    sorting : {
        boxSorting : '.box_sorting01',
        btnSorting : '.btn01',
        listSorting : '.list_sorting01',
        init : function(){
            var _this = this;
            _this.event();
        },
        event : function(){
            var _this = this;
            $(document).on('click', _this.btnSorting, function(e){
                e.preventDefault();
                if($(_this.btnSorting).closest(_this.boxSorting).find(_this.listSorting).is(':hidden')){
                    $(_this.btnSorting).addClass('on').attr('title','설정 상세 열기'); // 2021-06-18 Q10112 : on class 추가
                    $(_this.btnSorting).closest(_this.boxSorting).find(_this.listSorting).show();
                }else{
                    $(_this.btnSorting).removeClass('on').attr('title','설정 상세 닫기'); // 2021-06-18 Q10112 : on class 삭제
                    $(_this.btnSorting).closest(_this.boxSorting).find(_this.listSorting).hide();
                }
            });
        }
    },
    tableInit : {
        /*
        tableTarget : '.box_table table',
        init : function(){
            var _this = this;

            $(this.tableTarget).each(function(){
                var _this = $(this);
                tableCaption(_this)
            })
            function tableCaption(scope){
                var tableCaption = $(scope).find('> caption');
                var captionPElem = tableCaption.find('p');

                if(
                    (tableCaption.length > 0) &&
                    (captionPElem.length == 0 || $.trim(captionPElem.text())=='')

                ){
                    var msg='';
                    $(scope).find('> thead > tr >  th, > tbody > tr > th').each(function(){
                        var amsg=String($(this).clone().end().text() || '');
                        amsg=$.trim(amsg);

                        if($.trim(amsg)!=''){
                            msg += ((msg=='')?'':', ') + amsg;
                        }
                    });

                    captionPElem.remove();
                    $(document.createElement('p')).html(msg + '로 구성된 표입니다.').appendTo(tableCaption);
                };
            };
        }
        */
       tableTarget : '.box_table table',
        init : function(){

            $(this.tableTarget).each(function(){
                var _this = $(this);
                tableCaption(_this);
            })
            function tableCaption(scope){
                var slideLen = $(scope).closest('.swiper-slide').length;
                var popupLen = $(scope).closest('.layer_body').length;
                var formLen = $(scope).closest('.box_card01').length;
                var tableCaption = $(scope).find('> caption');
                var tableTitleEl = '';
                var summaryEl = '';
                var total = $(scope).find('> thead > tr >  th, > tbody > tr > th').length;
                if(popupLen === 1){//일반 팝업 테이블
                    tableTitleEl = $(scope).closest('.modal_pop').find('.layer_head').text();
                    $(scope).find('> thead > tr >  th, > tbody > tr > th').each(function(index){
                        if(index !== total - 1){
                            summaryEl += $(this).text() + ' ,';
                        } else {
                            summaryEl += $(this).text() + ' ';
                        }
                    });
                    tableTitle = tableTitleEl + ' - ' + summaryEl + '항목으로 구성된 표 입니다.';
                    tableCaption.html(tableTitle);
                }
                if(slideLen === 1){//슬라이드 형태 팝업 테이블
                    if ($(scope).prev("[class^='table_x_']").length) {
                        tableTitleEl = $(scope).prev("[class^='table_x_']").text();
                    } else if($(scope).prevAll("[class='tabletitle']:first").length) {
                        tableTitleEl = $(scope).prevAll("[class='tabletitle']:first").text();
                    } else if ($(scope).prevAll('[class^="h4_tit"]:first').length) {
                        tableTitleEl = $(scope).prevAll('[class^="h4_tit"]:first').text();
                    }  else if (!$(scope).prevAll('[class^="h4_tit"]:first').length && $(scope).prevAll('[class^="h3_tit"]:first').length) {
                        tableTitleEl = $(scope).prevAll('[class^="h3_tit"]:first').text();
                    } else if (!$(scope).prevAll('[class^="h4_tit"]:first').length && !$(scope).prevAll('[class^="h3_tit"]:first').length) {
                        if($(scope).closest('.box_layer').find('> [class^=type_]').length) {
                            tableTitleEl = $(scope).closest('.box_layer').find('> [class^=type_]').text();
                        } else {
                            tableTitleEl = $(scope).closest('.swiper-slide').find('.info_num').text();
                        }
                    } else {
                        tableTitleEl = $(scope).closest('.swiper-slide').find('.info_num').text();
                    }
                    tableTitle = tableTitleEl + ' - ' + summaryEl + '항목으로 구성된 표 입니다.';
                    tableCaption.html(tableTitle);
                }
                if(formLen === 1){//view table 형태로 변경
                    /*
                    $(scope).find('> thead > tr >  th, > tbody > tr > th').each(function(index){
                        if(index !== total - 1){
                            summaryEl += $(this).find('[class*="h4"]').text() + ' ,';
                        } else {
                            summaryEl += $(this).find('[class*="h4"]').text() + ' ';
                        }

                        if(index === 0){
                            tableTitleEl = $('.title_h72 h2.h3_b_lt').text();
                        } else {
                            tableTitleEl = $(scope).closest('.mt56').find('.title_h72 .h3_b_lt').text();
                        }
                    });
                    if($(scope).closest('.box_list_info_1').length === 1){
                        tableTitleEl = $(scope).closest('.box_list_info_1 > .h3_b_lt').text();
                    }
                    tableTitle = tableTitleEl + ' - ' + summaryEl + '항목으로 구성된 입력폼 입니다.';
                    tableCaption.html(tableTitle);
                    */
                   $(scope).find('> thead > tr >  th, > tbody > tr > th').each(function(){
                        $(this).wrap('<td></td>')
                        $(this).contents().unwrap();
                    });
                    $('.pd_t52 caption').remove();
                }


            };
        }
    },
    cardSvg: {
        init: function () {
            jQuery('img.svg').each(function () {
                var $img = jQuery(this);
                //var imgID = $img.attr('id');
                var imgClass = $img.attr('class');
                var imgURL = $img.attr('src');

                jQuery.get(imgURL, function (data) {
                    // Get the SVG tag, ignore the rest
                    var $svg = jQuery(data).find('svg');

                    // Add replaced image's ID to the new SVG
                    // if(typeof imgID !== 'undefined') {
                    //  $svg = $svg.attr('id', imgID);
                    // }
                    // Add replaced image's classes to the new SVG
                    if (typeof imgClass !== 'undefined') {
                        $svg = $svg.attr('class', imgClass + ' replaced-svg');
                    }

                    // Remove any invalid XML tags as per http://validator.w3.org
                    $svg = $svg.removeAttr('xmlns:a');

                    // Check if the viewport is set, else we gonna set it if we can.
                    if (!$svg.attr('viewBox') && $svg.attr('height') && $svg.attr('width')) {
                        $svg.attr('viewBox', '0 0 ' + $svg.attr('height') + ' ' + $svg.attr('width'))
                    }

                    // Replace image with new SVG
                    $img.replaceWith($svg);

                }, 'xml');

            });
        }
    },
    cardPop: {
        init: function () {
            var _this = this;
            _this.event($(this));
        },
        event: function ($this) {
            /* 팝업 tab */
            $('.modal_card_view .tab_default a').on('click', function (e) {
                e.preventDefault();

                $('.modal_card_view .tab_default a').removeClass('current');
                $(this).addClass('current');

                var activeTab = $(this).attr('href');
                $('.modal_card_view .tab_container').hide();
                $(activeTab).show();

                commonUi.scrbarUpdate.init($this.closest('.modal_pop').attr('id'), 'static');
            });

            if ($('.pop_premium_banner .swiper-slide').length > 1) {
                var cardEvent = new Swiper('.pop_premium_banner .swiper-container', {
                    loop: true,
                    navigation: {
                        prevEl: '.swiper-button-prev',
                        nextEl: '.swiper-button-next'
                    },
                    pagination: {
                        el: '.swiper-pagination',
                        clickable: true
                    },
                    observer: true,
                    observeParents: true,
                });
            } else {
                $('.pop_premium_banner .swiper-button-prev').hide();
                $('.pop_premium_banner .swiper-button-next').hide();
            }
        }
    }
}

/*
 *
 *  if(elemWrapper.find(".p3_m_lt_1ln:visible").length > 0){
        var messageArea = elemWrapper.find(".p3_m_lt_1ln:visible");
        var originText = elemWrapper.find(".p3_m_lt_1ln:visible").html();
        messageArea.attr("data-origin-text", originText).html(errorMsg).addClass("message_error");
    }else{
        var addErrorTag = $('<p class="p3_m_lt_1ln message_error"></p>').html(errorMsg);
        //element.closest("[class^='box_input']").append(addErrorTag);
        elemWrapper.append(addErrorTag);
    }

    elemContainer.addClass("error");

 */
//[HPRAGL-901]즉시결재 결제할 금액 1000원 미만으로 입력 : 에러메시지 출력시 focus 여부 선택 가능
var invalidMsg = {
    add : function(id, txt, error, focus){ // 에러메시지 출력시 focus 여부 선택 가능
        // error 메세지
        var focusCheck = focus == false ? focus : true; // 에러메시지 출력시 focus 여부 선택 가능
        var boxInpCellBox = $('#' + id).closest("[class^='box_input']");
        var messageArea = boxInpCellBox.find(".p3_m_lt_1ln:visible");
        if(messageArea.length > 0){
            var originText = messageArea.html();
            if ( !messageArea.attr("data-origin-text") && !messageArea.hasClass("msg_error") ) {
                messageArea.attr("data-origin-text", originText).html(txt).addClass("msg_error");
            } else {
                messageArea.html(txt).addClass("msg_error");
            }
        } else {
            boxInpCellBox.append($('<p class="p3_m_lt_1ln msg_error">' + txt + '</p>'));
        }
        // error 박스
        if(error == true){
            boxInpCellBox.find(".input_cell_box").addClass("error");
        }
        if(focusCheck == true){ // 에러메시지 출력시 focus 여부 선택 가능
            $('#' + id).focus();
        }
    },

    del : function(id, error){
        // error 메세지
        var boxInpCellBox = $('#' + id).closest("[class^='box_input']");
        var messageArea = boxInpCellBox.find(".p3_m_lt_1ln:visible");
        if(messageArea.attr("data-origin-text")){
            var originText = messageArea.attr("data-origin-text");
            messageArea.html(originText).removeClass("msg_error").removeAttr("data-origin-text");
        }else{
            if ( messageArea.hasClass("msg_error") ) {
                messageArea.remove();
            }
        }

        // error 박스
        if(error == false){
            boxInpCellBox.find(".input_cell_box").removeClass("error");
        }
    }

};

var popup = {
    open : function(id, btn, scroll, url){
        commonUi.layerPop.openLayer(id, btn, scroll, url);

    },
    close : function(id, btn){
        commonUi.layerPop.closeLayer(id, btn);
    }
}


/*
 * 키보드 트랩 : 레이어 팝업 tab, shift + tab 키 팝업 내 순환 처리
 */
var keyboardTrap = function(id, evt){
    var obj = $("#" + id);
    if ( evt.which == 9 ) {
        var o = obj.find('*');
        var focusableElems;
        focusableElems = o.filter(focusableElements).filter(':visible');
        var focusedElem;
        focusedElem = jQuery(':focus');
        var numOfFocusableElems;
        numOfFocusableElems = focusableElems.length;
        var focusedElemIdx;
        focusedElemIdx = focusableElems.index(focusedElem);
        if (evt.shiftKey) {
            if(focusedElemIdx==0){
                focusableElems.get(numOfFocusableElems-1).focus();
                evt.preventDefault();
            }
        } else {
            if(focusedElemIdx==numOfFocusableElems-1){
                focusableElems.get(0).focus();
                evt.preventDefault();
            }
        }
    }
    if (evt.which == 27){
        modal.close(id)
        evt.preventDefault();
    }
}

var modal = {
    open : function(id, btnid){
        var mdl = $("#" + id);
        returnFocusConfirm.push({pid : id, bid: btnid});
        if (mdl.hasClass('modal_alert')) {
            mdl.attr({
                'role': 'alert',
                'aria-modal':true
            });
        } else if (mdl.hasClass('modal_confirm')) {
            mdl.attr({
                'role': 'alertdialog',
                'aria-modal':true
            })
        }
        mdl.find(".layer_body").find("[class^='p1_']").eq(0).attr("id", id + "Title");
        mdl.attr("aria-labelledby", id + "Title");
        mdl.addClass("active").css("display","block").css("opacity",1).attr("tabindex",0).focus();
        mdl.on("keydown",function(event){
            keyboardTrap(id, event);
        });
    },
    close : function(id, btnid){
        var mdl = $("#" + id);
        var btn = $("#" + btnid);
        mdl.css("opacity",0).css("display","none").removeClass("active").attr("tabindex",-1).hide();//접근성 이슈로 hide 추가 2022-02-23 Q10011
        if(btnid){
            btn.focus();
        }else{
            $.each(returnFocusConfirm, function(index, value){
                if(value.pid == id) $("#" + value.bid).focus();
            });
        }
        $.each(returnFocusConfirm, function(index, value){
            if(value.pid == id) returnFocusConfirm.splice(index, 1);
        });
        mdl.off("keydown");
    }
}

var toggleCheckAll = function(elem){
    if(elem.find(".agree_list").length == 0){
        return;
    }
    var inpAll = elem.find(".box_chk01");
    var inpChild = elem.find(".agree_list");
    inpAll.on("change", "input[type='checkbox']", function(){
        if($(this).prop("checked") == true){
            inpChild.find("input[type='checkbox']").prop("checked", true);
            if (inpAll.hasClass('error')) {
                inpAll.removeClass('error');
                inpChild.find('.error').removeClass('error');
            }
        } else {
            inpChild.find("input[type='checkbox']").prop("checked", false);
        }
    });
    inpChild.on("change", "input[type='checkbox']", function(){
        if($(this).prop("checked") == false){
            inpAll.find("input[type='checkbox']").prop("checked", false);
        }else{
            if(inpChild.find("li:visible input[type='checkbox']:not(:checked)").length == 0){
                inpAll.find("input[type='checkbox']").prop("checked", true);
                inpAll.removeClass('error');
            }
        }
        if($(this).closest('.error').length) {
            $(this).closest('.error').removeClass('error');
        }
    });
};

/* GGCZ05 추가 - AS-IS 커스텀 셀렉트박스 및 의존 함수 */
var IEUA = (function(){
    var ua = navigator.userAgent.toLowerCase();
    var mua = {
            IE: /msie 8/.test(ua) || /msie 9/.test(ua) || /msie 7/.test(ua) || /msie 6/.test(ua) || /msie 5/.test(ua)
    };
    return mua;
})();
var _trans = true,
    _conScroll = false,
    _lastScroll = 0,
    _scrollTop = 0,
    _scrollDown = true,
    _topBtn = false,
    _hgt = $(window).height(),
_curFocus = null,
_curFocusObj = null;


// select option 추가
function optionAdd(target,val,text){
    var $target = $(target),
            _class='.select_wrap.' + target.split('#')[1];
    $target.append('<option value="' + val  + '">' + text + '</option>');
    $(_class).find('ul').append('<li><a href="#' + text + '">' + text + '</a></li>');
}
// select 박스 전체 추가
function selectAdd(target){
    var $target = this;
    if (target.length) $target = $(target);

    var herfTextDisable = $target.attr("herf-text-disable") || 'false';

    var $option = $target.find('option'),
            _size = $option.size(),
            _id = $target.attr('id'),
            _class = $target.attr('class'),
            _title = $target.attr('title'), //FIXME: GGCN45 2020-06-09 a 요소에 title 추가
            _focus = $target.find('option').attr('selected'), // 2022-01-04 Q10112 : selected 있는지 확인
            _val = '',
            _txt = '',
            _wrap = '',
            _arr = [],
            _selected = 0;
    if (_class == undefined){
            _class = '';
    }
    // 2022-01-04 Q10112 : selected 설정
    if (_focus == undefined){
        _focus = '';
    } else if(_focus ==  'selected'){
        _focus = 'on';
    }
    // 초기 selected
    for (var i = 0; i < _size; i++){
            _val = $option.eq(i).val();
            _txt = $option.eq(i).text();
            if ($option.eq(i).attr('selected') == 'selected'){
                    _selected = i;
            }
            _arr.push([_val,_txt]);
    }
    _wrap += '<div class="select_wrap ' +_id + ' ' + _class + '">';
    if (_selected == 0){
            if(_arr.length > 0){
                    _wrap += '<p><a href="#wrap" class="btn_select '+ _focus +'" role="button" title="'+ _title +'">' + _arr[0][1] + '</a></p>';  // 2022-01-04 Q10112 : selected 있는지 확인, FIXME: GGCN45 2020-06-09 a 요소에 title 추가
                } else {
                    _wrap += '<p><a href="#wrap" class="btn_select" role="button" title="'+ _title +'">' + "" + '</a></p>'; //FIXME: GGCN45 2020-06-09 a 요소에 title 추가
                }
    } else {
            _wrap += '<p><a href="#wrap" class="btn_select" role="button" title="'+ _title  +' 선택 값">' + _arr[_selected][1] + '</a></p>'; //FIXME: GGCN45 2020-06-09 a 요소에 title 추가
    }
    _wrap += '<div class="ul_select_list">';
    _wrap += '<ul class="ul_select" herf-text-disable="'+herfTextDisable+'">';
    for (var i in _arr){
            if(i != "valueIndex" && i != "removeDuplicates" && i != "indexOf") {
                if (i == _selected) _wrap += '<li class="on"><a href="#' + _arr[i][0] + '" title="' + _arr[i][1] + '"><span class="txt_select">' + _arr[i][1] + '</span><span class="blind">선택됨</span></a></li>'; // 2023-01-25 Q11067 #274 웹접근성 (선택여부)
                    else _wrap += '<li><a href="#' + _arr[i][0] + '" title="' + _arr[i][1] + '"><span class="txt_select">' + _arr[i][1] + '</span></a></li>'; // 2023-01-25 Q11067 #274 웹접근성 (선택여부)
            }
    }
    _wrap += '</ul">';
    _wrap += '</div">';
    _wrap += '</div">';
    $target.after(_wrap);

    if (_trans){
            $('.ul_select_list').mCustomScrollbar(
                {
                    horizontalScroll : false,
                    setTop: 0,
                    theme : 'minimal-dark',
                    mouseWheelPixels : 200, // 2022-01-14 Q10112 : 셀렉트 마우스 힐 수정
                    autoHideScrollbar : true,
                    advanced:{
                        updateOnContentResize: true
                    }
                }
            );
    } else {}
}
//select option 전체 교체
function selectChange(target){
    var $target = $(this);
    if (target) $target = $(target);
    var _class= '.select_wrap.' + target.split('#')[1],
            $ul = $(_class).find('.ul_select'),
            $option = $target.find('option'),
            _title = $target.attr('title'), //FIXME: GGCN45 2020-06-09 a 요소에 title 추가
            _size = $option.size(),
            _val = '',
            _txt = '',
            _wrap = '',
            _arr = [];
    if (_class == undefined){
            _class = '';
    }
    for (var i = 0; i < _size; i++){
            _val = $option.eq(i).val();
            _txt = $option.eq(i).text();
            _arr.push([_val,_txt]);
    }
    for (var i in _arr){
            if(i != "valueIndex" && i != "removeDuplicates") {
                    _wrap += '<li><a href="#' + _arr[i][0] + '"><span class="txt_select">' + _arr[i][1] + '</span></a></li>';
            }
    }

    $(_class).find('p').html('<a href="#wrap" class="btn_select" title="'+ _title  +' 선택 값">' + _arr[0][1] + '</a>'); //FIXME: GGCN45 2020-06-09 a 요소에 title 추가 
    $ul.html('').append(_wrap);
}

// 셀렉트 박스 - GGCZ05 : as-is에서 가져옴
var SELECT = (function(){
    var $select = $('select'),
            _winHgt = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight,
            _offset = 0,
            _cur = 0,
            _temp = false,
            _upTemp = false;

    $select.each(function(index, elemenent){
        // console.log(index)
            selectAdd($(this));
    })

    //카드이용내역 조회(사용처/기간 별) 기본 펼침
    if ( $( '.breakdown_inquery_section' ).length == 1 ){
            $( '.breakdown_inquery_section' ).find( '.select_wrap' ).addClass( 'spread' );
    }

    // 열기 버튼
    $(document).on('click','.btn_select',function(){
            if($(this).parent().closest('.form_select').hasClass('status')){

            }else{
            var $this = $(this),
                    $top = $this.closest('.select_wrap'),
                    _id = $top.attr('class'),
                    $ul = $top.find('.ul_select_list'),
                    _top = 0;
            var  _size = $ul.find('li').size(),
                    _liHgt = $ul.find('li').height(),
                    _ulHgt = _size * _liHgt;

                    // console.log("옵션갯수 : " + _size)
                    // console.log("옵션1개높이 : " + _liHgt)

            var btnHeight = $top.outerHeight();
            // disable 일경우
            if ($top.prev('select').attr('disabled')) return false;
            _offset = $top.offset().top - _scrollTop;
            closeSelect(_id);
            // 위로 아래로 열리는 방향 설정
            if (_winHgt - _offset < _winHgt / 2 && _ulHgt > _winHgt - _offset ) {
                    if (_trans){ // ie 8 제외 다른 브라우저
                            _upTemp = true;
                            $ul.addClass('up');
                    } else { // ie 8
                            _upTemp = false;
                            $ul.removeClass('up');
                    }
            } else {
                    _upTemp = false;
                    $ul.removeClass('up');
            }
            // 리사이즈시 높이값
            if (_winHgt < _offset + _ulHgt + 40 && !_upTemp ){
                    _top = _ulHgt - ((_offset + _ulHgt) - _winHgt + 50);
                    // console.log("높이값1");
            } else if ( _offset < _ulHgt && _upTemp ){
                    if (_winHgt - _offset > _ulHgt ) return;
                    _top = _offset - 20;
                    // console.log("높이값2");
            } else {
                _top = _ulHgt;
                // console.log("높이값3");
            }
            // console.log("높이값: " + _top)

            var mxH = 373;
            if($this.closest(".box_select").hasClass("h48")){
                mxH = 288;
            }
            /*
            var headerHeight = $("#header").height();
            if(_top > 500){
                    _top = _top-headerHeight;
            }*/
            if(_top > mxH){
                    _top = mxH;
            }

            // console.log("높이값 재설정 : " + _top)

            if ($top.hasClass('on')){
                    $ul.stop().animate({ height : '0' },200,function(){
                        $top.removeClass('on');
                        $ul.css({ display : 'none '});
                    });
                    if($this.closest('.form_select').hasClass('type_2')){
                        $this.closest('.form_select.type_2').stop().animate({ height : btnHeight}, 200);
                    }
                    _temp = false;
            } else {
                    // console.log("드롭다운 닫혀있을때")
                    $top.addClass('on');
                    if($this.closest('.form_select').hasClass('type_2')){
                        $ul.css({ height : 'auto' }).slideDown({
                            duration:200,
                            queue:false,
                            progress:function(){
                                $this.closest('.form_select.type_2').stop().animate({ height : btnHeight + $(this).find(".ul_select").outerHeight()}, 200);
                            }
                        });

                    }else{
                        $ul.css({ display : 'block' }).stop().animate({ height : _top + 'px' },200);
                    }
                    _temp = true;
            }
            return false;
    }})
    // 옵션 선택
    $(document).on('click','.ul_select a',function(){
        var $this = $(this),
        $top = $this.closest('.select_wrap'),
        $input = $top.find('.btn_select'),
        $ul = $top.find('.ul_select_list'),
        $openBtn = $top.find(' .btn_select'),
        $select = $top.prev('select'),
        _val = $this.val(),
        _txt = $this.find('.txt_select').text(), // 2023-01-25 Q11067 #274 웹접근성
        _idx = $this.parent().index();
        $ul.find('li').removeClass('on');
        $this.parent().addClass('on');
        // select 선택
        $select.find('option').attr('selected',false);
        $select.find('option').eq(_idx).attr('selected',true).prop('selected',true).change();

        // 상단 데이터 넣기
        var hrefTextDisable = $select.attr("herf-text-disable") || 'false';
        if (hrefTextDisable == 'false') {
            $input.attr('href',_val).text(_txt).addClass('on'); /* 2021-12-18 Q10112 : 셀렉트 선택했을때 볼드처리 추가 */
        }

        // 2023-01-25 Q11067 #274 웹접근성 (선택여부)
        var selectedTxt = '<span class="blind">선택됨</span>';
        $this.closest('.ul_select').find('.blind').remove();
        $this.append(selectedTxt);
        
        // 닫기
        if ($top.hasClass('spread')){
            $(this).css({ 'display':'block' });
        } else {
            closeSelect();
        }

        if($this.closest('.form_select').hasClass('type_2')){
            $this.closest('.form_select.type_2').stop().animate({ height : $top.outerHeight()}, 200);
        }
        $openBtn.focus();
        return false;
    })
    // 전체 닫기
    $(document).on('click',function(e){
        if ($('.select_wrap').hasClass('spread')){
            $(this).css({ 'display':'block' });
        } else {
            // 2016-12-02 셀렉트박스 스크롤바 클릭시 닫기 예외 추가
            if(!$(e.target).is('.mCSB_draggerRail') && !$(e.target).is('.mCSB_draggerContainer')  && !$(e.target).is('.mCS-dark')  && !$(e.target).is('.mCSB_container')  && !$(e.target).is('.mCSB_scrollTools_vertical')  && !$(e.target).is('.mCSB_dragger')) {
                closeSelect();
            }
        }
    })
    // 리사이즈
    if ($('select').length){
        $(window).resize(function(){
            _winHgt = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
            _offset = $select.offset().top;
        })
    };
    // 닫기 함수
    function closeSelect(other){
        var $target = $('.select_wrap'),
        _size = 0;
        _size = $('.select_wrap').size();
        for (var i = 0; i < _size; i++){
            var _id = $target.eq(i).attr('class');
            if (other === _id) continue;
            if($target.eq(i).closest('.form_select').hasClass('type_2')){
                $target.eq(i).closest('.form_select.type_2').removeAttr('style');
            }
            $target.eq(i).removeClass('on');
            $target.eq(i).find('.ul_select_list').css({ display : 'none' , height : '0' });
        }
        _temp = false;
    }
    $(document).on('focus, keydown','.select_wrap .btn_select',openKey);
    // 방향키로 선택
    function openKey(e){
        if (e.keyCode == 9) return; // 탭

        var $this = $(this),
        $ul = $this.closest('.select_wrap').find('.ul_select_list'),
        $li = $ul.find('li'),
        _size = $li.size(),
        _cur = 0;
        if (e.keyCode == 40 || e.keyCode == 38 || e.keyCode == 32 ){ // 화살표 아래, 위, 스페이스바 경우
            e.preventDefault();
        }
        if (_temp) return;
        if ($li.filter('.on').size() > 0){
            _cur = $li.filter('.on').index();
        }
        switch (e.keyCode){
            case 38:
            _cur--;
            break;
            case 40:
            _cur++;
            break;
        }
        if (_cur > _size - 1){
            _cur = _size - 1;
        } else if (_cur < 0){
            _cur = 0;
        }
        $li.eq(_cur).find('a').trigger('click');
    };
})();

// DatePicker - Q10071: as-is에서 가져옴
function initDatepicker(id){
    var el = id || document;
    var $el = $(el);

    $el.find(".datepicker-wrap").each(function(){
        var $that= $(this),
            $thatDateForamt= 'yymmdd',
            $buttonText = $that.find(">.vd-title").text();
        if($that.data("datepickerFormat")){
            $thatDateForamt = $that.data("datepickerFormat");
        }
        if($that.find("input.datepicker").length>1){
            $(this).find("input.datepicker").eq(0).datepicker({
                dateFormat : $thatDateForamt,
                showOn: "button",
                showOtherMonths : true,
                showMonthAfterYear : true,
                showButtonPanel: true,
                //yearSuffix : "년",
                dayNamesMin : ['일','월','화','수','목','금','토'],
                //monthNames : ['01월','02월','03월','04월','05월','06월','07월','08월','09월','10월','11월','12월'],
                monthNames : ['01','02','03','04','05','06','07','08','09','10','11','12'],
                buttonText : $buttonText+" 검색시작일 달력으로 선택<span></span>",
                onClose: function( selectedDate ) {
                    $that.find("input.datepicker").eq(1).datepicker( "option", "minDate", selectedDate );
                    $(this).removeClass("blank");
                }
            });
            widget = $that.datepicker( "widget" );
            //console.log(widget);
            $(this).find( "input.datepicker").eq(1).datepicker({
                dateFormat : $thatDateForamt,
                showOn: "button",
                showOtherMonths : true,
                showMonthAfterYear : true,
                showButtonPanel: true,
                //yearSuffix : "년",
                dayNamesMin : ['일','월','화','수','목','금','토'],
                //monthNames : ['01월','02월','03월','04월','05월','06월','07월','08월','09월','10월','11월','12월'],
                monthNames : ['01','02','03','04','05','06','07','08','09','10','11','12'],
                buttonText : $buttonText+"검색종료일 달력으로 선택<span></span>",
                onClose: function( selectedDate ) {
                    $that.find("input.datepicker").eq(0).datepicker( "option", "maxDate", selectedDate );
                    $(this).removeClass("blank");
                }
            });
        }else{
            $that.find("input.datepicker").eq(0).datepicker({
                dateFormat : $thatDateForamt,
                showOn: "button",
                showOtherMonths : true,
                showMonthAfterYear : true,
                showButtonPanel: true,
                //yearSuffix : "년",
                dayNamesMin : ['일','월','화','수','목','금','토'],
                //monthNames : ['01월','02월','03월','04월','05월','06월','07월','08월','09월','10월','11월','12월'],
                monthNames : ['01','02','03','04','05','06','07','08','09','10','11','12'],
                buttonText : $buttonText+" 달력으로 선택<span></span>"
            });
        }
    });
};

//Q10011 custom print 프린트 수정
var thisPrint =  function(idx){
    if($('#' + idx).hasClass('static')){
        $('#' + idx).find(".scrBarWrap").mCustomScrollbar('scrollTo' , 'top' , {scrollInertia : 0})
    }

    $('#' + idx).printThis({
        importCSS : false,
        importStyle : false,
        loadCSS : "/docfiles/resources/pc/css/print.css",
    });

    return false;

};



jQuery(document).ready(function(){
    commonUi.init();
    // 달력 팝업 초기화
    initDatepicker();

    $(document).find(".agree_wrap").each(function(){
        toggleCheckAll($(this));
    });

    /* var loc = window.location.href;   // 2022-05-23 GGCN45 : 목업 산출물 전용 코드, 운영 삭제
    if(loc.indexOf('/docfiles/resources/pc/html/') > -1 && loc.indexOf('.html') > -1 && loc.indexOf('PCCOM000000') < 0 && $("body").find("header").length > 0 && loc.indexOf('overallguide') < 0){
        $("body").append($('<div id="tempHeader" style="position:absolute; left:0; top:0; width:1px; height:1px; overflow:hidden; opacity:0"></div>'));
        $("body").append($('<div id="tempFooter" style="position:absolute; left:0; top:0; width:1px; height:1px; overflow:hidden; opacity:0"></div>'));

        $("#tempHeader").load('/docfiles/resources/pc/html/com/PCCOM000000.html header', function(){
            if(loc.indexOf('PCHOM000') < 0 && loc.indexOf('PCCOM000100') < 0 && loc.indexOf('guide') < 0 && loc.indexOf('PCCAR') < 0){
                $("body > .main > header").empty().html($("#tempHeader header").html());
            } else if(loc.indexOf('PCCAR') > -1 && loc.indexOf('PCCAR021900_external') < 0) {
                $("body .wrap > header").empty().html($("#tempHeader header").html());
            }
        });
        $("#tempFooter").load('/docfiles/resources/pc/html/com/PCCOM000000.html footer', function(){
            if(loc.indexOf('PCHOM000') < 0 && loc.indexOf('PCCAR') < 0){
                $("body > .main > footer").empty().html($("#tempFooter footer").html());
            } else if(loc.indexOf('PCCAR') > -1) {
                $("body .wrap > footer").empty().html($("#tempFooter footer").html());
            }
        });
    } */



    // 임시  카드상세
    //가로형일때
    $(document).find('.card_detail_view .detail_top .left_area .img_wrap.horizontal').each(function(){
        $(this).removeClass('horizontal').closest('.detail_top').addClass('horizontal');
    });

    //Q10048 고객지원 검색영역 추가
    $('.box_search input').on('input',function () {
        if ($(this).val()=="") {
            $('.box_search .btn_del').hide();
        } else {
            $('.box_search .btn_del').show();
        }
    });

    $('.box_search .btn_del').on('click', function () {
        $(this).siblings('input').val('');
        $(this).hide();
    });

    /* 고객지원 textarea */
    $(document).find(".input_textarea").each(function () {
        var textA = $(this).find('textarea');
        textA.on('focus', function () {
            $(this).closest('.input_textarea').addClass('on');
        });

        textA.on('focusout', function () {
            $(this).closest('.input_textarea').removeClass('on');
        });

    });

});


var initUiCardBill = function(){
    /*
    var divHistory = document.getElementById('divHistoryUse');
    var divTop = '';

    window.addEventListener('scroll', function(){
        divTop = window.pageYOffset + divHistory.getBoundingClientRect().top - 48;
        console.log("rect" + divTop)
        console.log("scr" + window.pageYOffset)
        if(divTop < window.pageYOffset){
            $("#divHistoryUseLeft").css("position","fixed").css("top","48px");
        }else{
            $("#divHistoryUseLeft").css("position","").css("top","");
        }
    });
    */
}

// 탭 속성 변경 및 삭제 - 웹 접근성 심사의 role 삭제로 인해 추가 2022-04-06
function tabAttrChange() {
    var $tab = $('[role^="tab"]').not('[role*="tablist"], [role*="tabpanel"]');
    var $tablist = $('[role*="tablist]"');
    var $tabpanel = $('[role="tabpanel"]');
    $tab.each(function(i){
        var _this = $(this);
        var _role = _this.attr('role');
        var _location = window.location.href;
        var target = _this.attr('aria-controls');
        var hash = _this.attr('href') ? _this.attr('href') : ""; //2022-05-27 : href가  없는 경우
        var tabListId = "#cardTab";
        var tabList = _this.closest(tabListId);
        var tabText = _this.text();
        if(hash.indexOf('.hc') === -1 && _location.indexOf('CPUUG3001_01') === -1 && _location.indexOf('CPACB0201_01') === -1){ //href에 링크 주소가 없을경우만 실행 , 뉴스.공지(CPUUG3001_01), 포인트(CPACB0201_01) 전체 예외처리
            if(tabList.length) { //카드 서브메인
                $('#' + target).prepend('<h2 class="a11y">' + tabText +'</h2>');
            } else {
                if($('#' + target).find('h4:not(.a11y)').length) {
                    $('#' + target).find('h4').before('<h3 class="a11y">' + tabText +'</h3>');
                } else if($('#' + target).find('h3:not(.a11y)').length) {
                    $('#' + target).find('h3').after('<h4 class="a11y">' + tabText +'</h4>');
                } else {
                    $('#' + target).prepend('<h4 class="a11y">' + tabText +'</h4>');
                }
            }
        }
        _this.removeAttr('tabindex role aria-selected').attr('data-role', _role);
    });
    // 로그인 페이지 별도 적용
    if($('.sub_login ').length){
        $('.tab').find('li > a').each(function(){
            var target = $(this).attr('href');
            $(target).prepend('<h3 class="a11y">'+ $(this).text() +'</h3>');
        })
    }
    $tablist.each(function(){
        var _this = $(this);
        var _role = _this.attr('role');
        _this.removeAttr('role').attr('data-role', _role);
    });
    $tabpanel.each(function(){
        $(this).removeAttr('role').attr('data-role', 'tabpanel');
    });

}

$(window).load(function () {
    commonUi.gnb.gnbHeader();
    if($('img.svg').length) { commonUi.cardSvg.init() }
    if($('.accodWrap').length) { commonUi.accodiran.init() }
    if($('.modal_pop').length) {
        commonUi.layerPop.init(); $('.modal_pop').hide();
    } else {
        setTimeout(function(){
            if($('.modal_pop').length) {
                commonUi.layerPop.init(); $('.modal_pop').hide();
            }
        }, 800);
    }//접근성 이슈로 hide 추가 2022-02-23 Q10011
    if($('.box_input01').length) { commonUi.inp.init(); commonUi.deleteTxt.init(); }
    if($('[role^="tab"]').length) {
        tabAttrChange();
    } else {
        setTimeout(function(){
            if($('[role^="tab"]').length) {
                tabAttrChange();
            }
        }, 800);
    }
    ttCont();
    inputSetProperty();
});

//Q10011 개발팀 요청에 해당 스크립트 추가 2022-01-11
function inputSetProperty(target){
    var _target = target ? target : $('input');
    var inputValProp = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
    inputValProp.set = function(value) {
        var orgInputValProp = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
        Object.defineProperty(this, 'value', {
            set : orgInputValProp.set
        });
        this.value = value;
        setInputAction(this);
        Object.defineProperty(this, 'value', inputValProp);
    }

    _target.map(function(i, o) {
        if (/text|tel/.test(this.type)) {
            return o;
        }
    }).each(function() {
        setInputAction(this);
        Object.defineProperty(this, 'value', inputValProp);
    });
}

//Q10011 개발팀 요청에 해당 스크립트 추가 2022-01-11
function setInputAction(_this) {
    var inputCellBox = $(_this).closest('.input_cell_box');
    if (inputCellBox.size()) {
        if (_this.value) {
            if (!inputCellBox.hasClass('completed')) {
                inputCellBox.addClass('completed');
            }
        } else {
            if (inputCellBox.hasClass('completed')) {
                inputCellBox.removeClass('completed');
            }
            if (inputCellBox.hasClass('on')) {
                inputCellBox.removeClass('on');
            }
            if (inputCellBox.hasClass('focused') && !$(_this).is(':focus')) {
                inputCellBox.removeClass('focused');
            }
        }
    }
}

// Q10473 :  웹접근성 익스플로러 키보드 초점 스크립트 2022-01-25

function applyFocusVisiblePolyfill(scope) {
    var hadKeyboardEvent = true;
    var hadFocusVisibleRecently = false;
    var hadFocusVisibleRecentlyTimeout = null;

    var inputTypesAllowlist = {
      text: true,
      search: true,
      url: true,
      tel: true,
      email: true,
      password: true,
      number: true,
      date: true,
      month: true,
      week: true,
      time: true,
      datetime: true,
      'datetime-local': true
    };

    /**
     * Helper function for legacy browsers and iframes which sometimes focus
     * elements like document, body, and non-interactive SVG.
     * @param {Element} el
     */
    function isValidFocusTarget(el) {
      if (
        el &&
        el !== document &&
        el.nodeName !== 'HTML' &&
        el.nodeName !== 'BODY' &&
        'classList' in el &&
        'contains' in el.classList
      ) {
        return true;
      }
      return false;
    }

    /**
     * Computes whether the given element should automatically trigger the
     * `focus-visible` class being added, i.e. whether it should always match
     * `:focus-visible` when focused.
     * @param {Element} el
     * @return {boolean}
     */
    function focusTriggersKeyboardModality(el) {
      var type = el.type;
      var tagName = el.tagName;

      if (tagName === 'INPUT' && inputTypesAllowlist[type] && !el.readOnly) {
        return true;
      }

      if (tagName === 'TEXTAREA' && !el.readOnly) {
        return true;
      }

      if (el.isContentEditable) {
        return true;
      }

      return false;
    }

    /**
     * Add the `focus-visible` class to the given element if it was not added by
     * the author.
     * @param {Element} el
     */
    function addFocusVisibleClass(el) {
      if (el.classList.contains('focus-visible')) {
        return;
      }
      el.classList.add('focus-visible');
      el.setAttribute('data-focus-visible-added', '');
    }

    /**
     * Remove the `focus-visible` class from the given element if it was not
     * originally added by the author.
     * @param {Element} el
     */
    function removeFocusVisibleClass(el) {
      if (!el.hasAttribute('data-focus-visible-added')) {
        return;
      }
      el.classList.remove('focus-visible');
      el.removeAttribute('data-focus-visible-added');
    }

    /**
     * If the most recent user interaction was via the keyboard;
     * and the key press did not include a meta, alt/option, or control key;
     * then the modality is keyboard. Otherwise, the modality is not keyboard.
     * Apply `focus-visible` to any current active element and keep track
     * of our keyboard modality state with `hadKeyboardEvent`.
     * @param {KeyboardEvent} e
     */
    function onKeyDown(e) {
      if (e.metaKey || e.altKey || e.ctrlKey) {
        return;
      }

      if (isValidFocusTarget(scope.activeElement)) {
        addFocusVisibleClass(scope.activeElement);
      }

      hadKeyboardEvent = true;
    }

    /**
     * If at any point a user clicks with a pointing device, ensure that we change
     * the modality away from keyboard.
     * This avoids the situation where a user presses a key on an already focused
     * element, and then clicks on a different element, focusing it with a
     * pointing device, while we still think we're in keyboard modality.
     * @param {Event} e
     */
    function onPointerDown(e) {
      hadKeyboardEvent = false;
    }

    /**
     * On `focus`, add the `focus-visible` class to the target if:
     * - the target received focus as a result of keyboard navigation, or
     * - the event target is an element that will likely require interaction
     *   via the keyboard (e.g. a text box)
     * @param {Event} e
     */
    function onFocus(e) {
      // Prevent IE from focusing the document or HTML element.
      if (!isValidFocusTarget(e.target)) {
        return;
      }
      if($(e.target).parent("div").attr("class") == 'user-skip'){
        actionKeyboardFocus = true;
      }
      if (hadKeyboardEvent || focusTriggersKeyboardModality(e.target)) {
        addFocusVisibleClass(e.target);
      }
    }

    /**
     * On `blur`, remove the `focus-visible` class from the target.
     * @param {Event} e
     */
    function onBlur(e) {
      if (!isValidFocusTarget(e.target)) {
        return;
      }

      if (
        e.target.classList.contains('focus-visible') ||
        e.target.hasAttribute('data-focus-visible-added')
      ) {
        // To detect a tab/window switch, we look for a blur event followed
        // rapidly by a visibility change.
        // If we don't see a visibility change within 100ms, it's probably a
        // regular focus change.
        hadFocusVisibleRecently = true;
        window.clearTimeout(hadFocusVisibleRecentlyTimeout);
        hadFocusVisibleRecentlyTimeout = window.setTimeout(function() {
          hadFocusVisibleRecently = false;
        }, 100);
        removeFocusVisibleClass(e.target);
      }
    }

    /**
     * If the user changes tabs, keep track of whether or not the previously
     * focused element had .focus-visible.
     * @param {Event} e
     */
    function onVisibilityChange(e) {
      if (document.visibilityState === 'hidden') {
        // If the tab becomes active again, the browser will handle calling focus
        // on the element (Safari actually calls it twice).
        // If this tab change caused a blur on an element with focus-visible,
        // re-apply the class when the user switches back to the tab.
        if (hadFocusVisibleRecently) {
          hadKeyboardEvent = true;
        }
        addInitialPointerMoveListeners();
      }
    }

    /**
     * Add a group of listeners to detect usage of any pointing devices.
     * These listeners will be added when the polyfill first loads, and anytime
     * the window is blurred, so that they are active when the window regains
     * focus.
     */
    function addInitialPointerMoveListeners() {
      document.addEventListener('mousemove', onInitialPointerMove);
      document.addEventListener('mousedown', onInitialPointerMove);
      document.addEventListener('mouseup', onInitialPointerMove);
      document.addEventListener('pointermove', onInitialPointerMove);
      document.addEventListener('pointerdown', onInitialPointerMove);
      document.addEventListener('pointerup', onInitialPointerMove);
      document.addEventListener('touchmove', onInitialPointerMove);
      document.addEventListener('touchstart', onInitialPointerMove);
      document.addEventListener('touchend', onInitialPointerMove);
    }

    function removeInitialPointerMoveListeners() {
      document.removeEventListener('mousemove', onInitialPointerMove);
      document.removeEventListener('mousedown', onInitialPointerMove);
      document.removeEventListener('mouseup', onInitialPointerMove);
      document.removeEventListener('pointermove', onInitialPointerMove);
      document.removeEventListener('pointerdown', onInitialPointerMove);
      document.removeEventListener('pointerup', onInitialPointerMove);
      document.removeEventListener('touchmove', onInitialPointerMove);
      document.removeEventListener('touchstart', onInitialPointerMove);
      document.removeEventListener('touchend', onInitialPointerMove);
    }

    /**
     * When the polfyill first loads, assume the user is in keyboard modality.
     * If any event is received from a pointing device (e.g. mouse, pointer,
     * touch), turn off keyboard modality.
     * This accounts for situations where focus enters the page from the URL bar.
     * @param {Event} e
     */
    function onInitialPointerMove(e) {
      // Work around a Safari quirk that fires a mousemove on <html> whenever the
      // window blurs, even if you're tabbing out of the page. ¯\_(ツ)_/¯
      if (e.target.nodeName && e.target.nodeName.toLowerCase() === 'html') {
        return;
      }

      hadKeyboardEvent = false;
      removeInitialPointerMoveListeners();
    }

    // For some kinds of state, we are interested in changes at the global scope
    // only. For example, global pointer input, global key presses and global
    // visibility change should affect the state at every scope:
    document.addEventListener('keydown', onKeyDown, true);
    document.addEventListener('mousedown', onPointerDown, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('touchstart', onPointerDown, true);
    document.addEventListener('visibilitychange', onVisibilityChange, true);

    addInitialPointerMoveListeners();

    // For focus and blur, we specifically care about state changes in the local
    // scope. This is because focus / blur events that originate from within a
    // shadow root are not re-dispatched from the host element if it was already
    // the active element in its own scope:
    scope.addEventListener('focus', onFocus, true);
    scope.addEventListener('blur', onBlur, true);

    // We detect that a node is a ShadowRoot by ensuring that it is a
    // DocumentFragment and also has a host property. This check covers native
    // implementation and polyfill implementation transparently. If we only cared
    // about the native implementation, we could just check if the scope was
    // an instance of a ShadowRoot.
    if (scope.nodeType === Node.DOCUMENT_FRAGMENT_NODE && scope.host) {
      // Since a ShadowRoot is a special kind of DocumentFragment, it does not
      // have a root element to add a class to. So, we add this attribute to the
      // host element instead:
      scope.host.setAttribute('data-js-focus-visible', '');
    } else if (scope.nodeType === Node.DOCUMENT_NODE) {
      document.documentElement.classList.add('js-focus-visible');
      document.documentElement.setAttribute('data-js-focus-visible', '');
    }
  }

  // It is important to wrap all references to global window and document in
  // these checks to support server-side rendering use cases
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    // Make the polyfill helper globally available. This can be used as a signal
    // to interested libraries that wish to coordinate with the polyfill for e.g.,
    // applying the polyfill to a shadow root:
    window.applyFocusVisiblePolyfill = applyFocusVisiblePolyfill;

    // Notify interested libraries of the polyfill's presence, in case the
    // polyfill was loaded lazily:
    var event;

    try {
      event = new CustomEvent('focus-visible-polyfill-ready');
    } catch (error) {
      // IE11 does not support using CustomEvent as a constructor directly:
      event = document.createEvent('CustomEvent');
      event.initCustomEvent('focus-visible-polyfill-ready', false, false, {});
    }

    window.dispatchEvent(event);
  }

  if (typeof document !== 'undefined') {
    // Apply the polyfill to the global document, so that no JavaScript
    // coordination is required to use the polyfill in the top-level document:
    applyFocusVisiblePolyfill(document);
  }


  // Q10473 포커스가 화면 밖으로 사라졌을 때 자동스크롤
  function ChekPageFocus() {
      if(!document.hasFocus()) {
        $('html, body').animate({scrollTop:$(window).scrollTop()},0);
      }
  }
ChekPageFocus();

$(function () {
    // Q10473 달력 레이어 자동 포커스
    setTimeout(function () {
        $(".datepicker-wrap input[type='text']").on('focus focusin', function (event) {
            if (!actionKeyboardFocus) {
                $(".ui-datepicker-hc-container .ui-datepicker-hc-holder .ui-datepicker-hc-today .ui-state-highlight").focus();
                $(this).removeClass('focus-visible');
                $(this).removeAttr('data-focus-visible-added');
                $(this).blur();
            } else if ($(this).hasClass('hasDatepickerHC')) {
                var titTxt = $(this).attr('title');
                var phText = $(this).attr('placeholder').toUpperCase(); // 2022. 1. 1 타입이슈로 추가
                var addTxt;
                if(phText === 'YYYY. M. D') {
                    addTxt = ' (2022. 1. 1 형태로 입력해주세요. .뒤에 공백이 있어야합니다.)';
                } else {
                    addTxt = ' (22. 1. 1 형태로 입력해주세요. .뒤에 공백이 있어야합니다.)';
                }
                if ($(this).attr('data-title-apply') === undefined && $(this).attr('data-title-apply') !== true) { //반복 텍스트 적용을 방지
                    $(this).attr({ 'title': titTxt + addTxt, 'data-title-apply':true });
                }
                $(this).removeAttr('readonly');
            }
        })
    }, 0);
})

/*
1. 대화상자를 여는 버튼에는 aria-haspopup="dialog" 속성과 aria-controls 속성을 줍니다. aria-controls 에는 버튼과 연결된 role="dialog" 요소의 id 를 주면 됩니다.
예: <button aria-haspopup="dialog" aria-controls="dialog-container">대화상자 열기</button>
2. role="dialog" 속성은 대화상자가 있는 컨테이너에 주되 display 속성이 none, block 으로 변경되는 곳이어야 합니다.
3. aria-haspopup="dialog" 를 클릭하여 대화상자가 열렸을 때 초점을 대화상자 내부의 특정 요소로 보내려면 보내고자 하는 요소에 autoFocus 라는 class 를 주면 됩니다. 다만 탭키로 접근이 가능한 요소이거나, 자바스크립트로 초점을 보낼 수 있도록 tabindex="-1" 속성이 들어가 있는 요소여야 합니다.
4. 대화상자 내부에서 포커스 트랩을 구현하기 위해 class="firstTab" class="lastTab" 클래스를 각각 지정합니다. 이렇게 하면 firstTab 요소에서 쉬프트 탭을 누르면 lastTab class 로, lastTab class 에서 탭키를 누르면 firstTab class 로 이동합니다.
5. 대화상자를 닫는 버튼에는 class="modalClose" 를 추가합니다. 그러면 취소 키를 눌렀을 때 해당 요소가 클릭되면서 대화상자가 사라지고 초점은 이전 대화상자를 여는 버튼으로 돌아가게 됩니다. 대화상자가 display:none 되면 모든 aria-hidden 속성은 사라집니다.


'use strict';

var $body = document.body,
    $targetAreas = $body.querySelectorAll('[aria-haspopup=dialog]'),
    modals = $body.querySelectorAll('[role=dialog], [role=alertdialog]'),
    $modal = null,
    $firstTab,
    $lastTab,
    $closeModal,
    $targetArea;
$targetAreas.forEach(function ($el) {
    $el.addEventListener('click', initialize, false);
});
*/
function initialize(event) {
    setTimeout(function () {
        $targetArea = event.target;
        modals.forEach(function ($el) {
            if ($targetArea.getAttribute('aria-controls') && $targetArea.getAttribute('aria-controls') == $el.getAttribute('id') && 'true' == $el.getAttribute('aria-modal') && window.getComputedStyle($el).display === "block" || $el.getAttribute('aria-hidden') === 'false') {
                $modal = $el;
                if ($modal.querySelector(".autoFocus")) {
                    $modal.querySelector(".autoFocus").focus();
                }
            }
        });

        if ($modal) {
            (function () {
                $closeModal = $modal.querySelector('.closeModal'), $firstTab = $modal.querySelector('.firstTab'), $lastTab = $modal.querySelector('.lastTab');
                setHiddenExceptForThis($modal);
                if (!$modal.getAttribute('aria-label') || $modal.getAttribute('aria-labelledby')) {
                    $modal.setAttribute('aria-label', $targetArea.textContent);
                }
                $modal.addEventListener('keydown', bindKeyEvt);
                var observer = new MutationObserver(function (mutations) {
                    setHiddenExceptForThis($modal, 'off');
                    setTimeout(function () {
                        if (window.getComputedStyle($modal).display === "none" || $modal.getAttribute('aria-hidden') === 'true') {
                            $targetArea.focus();
                            $modal.removeEventListener("keydown", bindKeyEvt, false);
                            observer.disconnect();
                        }
                    }, 500);
                });
                var option = {
                    attributes: true,
                    CharacterData: true
                };
                observer.observe($modal, option);
            })();
        }
    }, 500);
}

function bindKeyEvt(event) {
    event = event || window.event;
    var keycode = event.keycode || event.which;
    var $target = event.target;

    switch (keycode) {
        case 9:
            // tab key
            if ($firstTab && $lastTab) {
                if (event.shiftKey) {
                    if ($firstTab && $target == $firstTab) {
                        event.preventDefault();
                        if ($lastTab) $lastTab.focus();
                    }
                } else {
                    if ($lastTab && $target == $lastTab) {
                        event.preventDefault();
                        if ($firstTab) $firstTab.focus();
                    }
                }
            } else {
                event.preventDefault();
            }
            break;
        case 27:
            // esc key
            event.preventDefault();
            $closeModal.click();
            break;
        default:
            break;
    }
}

/*
1. element 파라미터에는 role="dialog"가 붙은 컨테이너를 document.querySelector()나 document.getElementById()등으로 가져와서 넣습니다.
2. turn은 'on'과 'off'값이 허용되며, on이면 element로 지정된 요소가 속한 부모 요소들과 element의 하위 요소, 그리고 element 자신을 제외한 모든 요소에 aria-hidden="true"를 추가해 줍니다.
3. 이 함수로 aria-hidden="true" 가 부여된 요소는 is-sr-hidden 서브클래스가 붙으며, 같은 요소에 'off'를 사용하여 이 함수를 다시 부르면 aria-hidden 속성이 제거됩니다.
*/

function setHiddenExceptForThis(element) {
    var turn = arguments.length <= 1 || arguments[1] === undefined ? 'on' : arguments[1];


    // 다른 라이브러리로 인해 aria-hidden이 추가된 요소를 제외한 모든 요소를 가져옵니다. (버그 방지를 위해 aria-hidden이 없는 요소만을 가져옵니다)
    var allElems = document.body.querySelectorAll('*:not([aria-hidden="true"])');

    // 혹시 모를 버그를 방지하기 위해 aria-hidden을 초기화합니다.
    allElems.forEach(function (el) {
        el.removeAttribute('aria-hidden');
    });

    // Array.from과 같은 간단한 방법으로 Array로 바꿀 수 있으나 호환성 이슈로 NodeList에서 Array로 바꾸는 작업에 반복문을 사용합니다.
    var _allElems = [];
    for (var i = 0; i < allElems.length; i++) {
        _allElems.push(allElems[i]);
    }

    // 숨겨질, 중요하지 않은 요소들과 그렇지 않은 대화상자 요소를 걸러내어, 대화상자와 관계없는 요소들을 모두 추려냅니다.
    var notImportants = _allElems.filter(function (el) {
        if (element.contains(el) === false && el.contains(element) === false) {
            return el;
        }
    });

    // 'on'일 때 notImportants안에 들어있는 요소들을 모두 aria-hidden="true" 처리하고, is-sr-hidden 클래스를 추가합니다.
    if (turn === 'on') {
        notImportants.forEach(function (el) {
            el.setAttribute('aria-hidden', 'true');
            el.classList.add('is-sr-hidden');
        });
    }

    // 'off'일 때 'is-sr-hidden'클래스를 가진 요소 목록을 가져와서 aria-hidden과 식별용 is-sr-hidden 클래스를 제거합니다.
    if (turn === 'off') {
        document.querySelectorAll('.is-sr-hidden').forEach(function (el) {
            el.classList.remove('is-sr-hidden');;
            el.removeAttribute('aria-hidden');
        });
    }
}

//Q10011 웹접근성 ie 초점이동 관련 js
$(document).on('focus' , '.sec_main_login .sec_right_head a' , function(){ // 2023-05-09 GGCP31 로그인 영역에만 해당되게 .sec_main_login 클래스 추가
    $('html , body').scrollTop(0);
});
$(document).ready(function() {
    $("#savingContent .textbico_mide").one('focusin' , function(){ // 2023-05-04 GGCP31 .right_content 제거, .account .fr -> textbico_mide 로 수정
        document.documentElement.scrollTop = 0;
    });
});
//팝업 닫기 클릭시에 포커스 이동 추가
$(document).on('click' , 'a' , function(){
	var _this = $(this); //2022-05-09 GGCN45: 전역변수 오염 (var 추가)
    var targetEvent = _this.attr('href') ? _this.attr('href').indexOf('popup.open') : -1;//팝업 open 버튼 유무 확인  //2022-05-09 GGCN45: 전역변수 오염 (var 추가)
	if(_this[0]['onclick'] !== null){//타겟에 onclick 속성이 있는경우 targetEvent 값 변경
		targetEvent = _this.attr('onclick').indexOf('popup.open');
		//console.log('targetEvent 값 변경' , targetEvent)
	}
	if(targetEvent !== -1){
		_this.attr('data-popup', 'true');
	}
});

//popup loop
$(document).on('keydown' , '.modal_pop' , function(e){
	_this = $(this);
	focuEl = _this.find('.layer_wrap *').find(focusableElements);
	if(e.keyCode === 9){
		focusItem = $(':focus');

		if(e.shiftKey){//shift key
			if(focuEl.index(focusItem) === 0){
				focuEl.eq(focuEl.length - 1).focus();
				e.preventDefault();
			}
		} else {
			if(focuEl.index(focusItem) === focuEl.length - 1){
				focuEl.eq(0).focus();
				e.preventDefault();
			}
		}
	}
	/*
	_this = $(this);
	if(event.which == 9){
		var elems = _this.find("*");
		var focusableItems;
		focusableItems = elems.filter(focusableElements).filter(":visible");
		var focusedItem;
		focusedItem = $(":focus");
		var numFocusableItems;
		numFocusableItems = focusableItems.length;
		var focusedItemIndex;
		focusedItemIndex = focusableItems.index(focusedItem);
		if(event.shiftkey){
			if(focusedItemIndex == 0){
				focusableItems.get(numFocusableItems - 1).focus();
				event.preventDefault();
			}
		}else{
			if(focusedItemIndex == numFocusableItems - 1){
				focusableItems.get(0).focus();
				event.preventDefault();
			}
		}
	}
	*/
});


//접근성 상단 fix영역으로 인해 콘텐트가 가려지는 현상 수정
$(window).on('hashchange' , function(){
    var targetHash = window.location.hash;
    if(targetHash === '#container'){
        $('#container').css('padding-top','96px');
    }
    $(window).on('scroll' , function(){
        var winScrollTop = $(window).scrollTop();
        if(winScrollTop >= 192){
            $('#container').css('padding-top','');
            $(window).off('scroll');
        }
    })
});


/* 2022-05-27 GGCN45 : Common Tab Fn : mobile commonUi.js 참고    [고객지원]소비자보호 하위 탭 추가 및 컨텐츠 수정 요청의 건(220602)*/
;(function(window, document, $) {
    "use strict";
    // 전체 공통 탭 메뉴
    $.fn.tabMenus = function(options) {
        var settings = $.extend({}, $.fn.tabMenus.defaults, options || {});
        var self = this;

        return self.each(function() {
            self.$selector = $(this);
            self.$menu = self.$selector.find('> .' + settings.tabMenuClass);
            self.$contents = self.$selector.find('.' + settings.tabContsClass);
            self.$activate = settings.activeClass;
            self._eAction = settings.event;

            self._create = function() { // 기본세팅
                $(self.$contents).css('display', 'none');
                self.$menu.attr('role', 'tablist');
                self.$menu.find('> li').each(function() {
                    var _this = $(this);
                    if (!_this.find('a').length) {
                        return
                    }
                    var str = /\#/gi;
                    var _anchor = _this.find('a').attr('href');

                    _this.attr({
                        'id': _anchor.replace(str, 'anchor-'),
                        'role': 'tab',
                        // 'tabindex': 0,
                        'aria-selected': false,
                        'aria-expanded': false
                    }).find('a').attr({
                        'role': 'presentation',
                        // 'tabindex': -1
                    }).addClass('tabs-anchor');
                });
                self.$contents.each(function(i) {
                    var _this = $(this);
                    _this.attr({
                        'role': 'tabpanel',
                        'aria-hidden': true,
                        'aria-labelledby': self.$menu.find('> li').eq(i).attr('id')
                    });
                });

                self._isLocal();
            };

            self._isLocal = function() { //재설정
                var elem;
                if (settings.startItem > 1) {
                    elem = self.$menu.find('> li:nth-child(' + settings.startItem + ') ').find('a').attr('href');

                    self.$menu.find('.' + self.$activate).attr({
                        'aria-selected': false,
                        'aria-expanded': false
                    }).removeClass(self.$activate);
                    self.$menu.find('> li:nth-child(' + settings.startItem + ') ').attr({
                        // 'tabindex': 0,
                        'aria-selected': true,
                        'aria-expanded': true
                    }).find('a').addClass(self.$activate);
                    $(elem).css('display', 'block').attr('aria-hidden', false);
                } else {
                    elem = self.$menu.find('> li:first').find('a').attr('href');

                    self.$menu.find('> li:first').attr({
                        // 'tabindex': 0,
                        'aria-selected': true,
                        'aria-expanded': true
                    }).find('a').addClass(self.$activate).attr('title', '선택됨'); // Q10474 WAI title 추가
                    $(elem).css('display', 'block').attr('aria-hidden', false);
                }

                self.Action();
            };

            self.Action = function() {
                self.$menu.on(self._eAction, 'a', function(e) {
                    var _this = $(this);

                    if (!_this.hasClass(self.$activate)) {
                        _this.addClass(self.$activate).closest('li').attr({
                            // 'tabindex': 0,
                            'aria-selected': true,
                            'aria-expanded': true
                        }).siblings().attr({
                            // 'tabindex': -1,
                            'aria-selected': false,
                            'aria-expanded': false
                        }).find('.' + self.$activate).removeClass(self.$activate).attr('title', '');
                        _this.addClass(self.$activate).closest('li').find('a').attr('title', '선택됨'); // Q10474 WAI title 추가
                        if ($(_this.attr('href')) !== '#' || $(_this.attr('href')) !== '#none' || $(_this.attr('href')) !== '') {
                            $(_this.attr('href')).css('display', 'block').attr('aria-hidden', false).siblings('div' + ('.' + settings.tabContsClass)).css('display', 'none').attr('aria-hidden', true);
                        }
                    }

                    return false;
                });
            };

            self._init = function() {
                if (!self.$menu.length) {
                    return;
                }
                self._create();
            };


            self._init();
        });
    };


    $.fn.tabMenus.defaults = {
        startItem: 1, // 먼저 보여줄 탭
        tabMenuClass: 'ui_tab_menu', // 탭 메뉴 기본 클래스
        tabContsClass: 'ui_tab_container', // 탭 패널 기본 클래스
        activeClass: 'current', //활성화 클래스
        event: 'click' //mouseenter, mouseover
    };
})(window, document, jQuery);

/* common utils */
function getParamsFromUrl(url){ //2022-06-03 GGCN45 : 공통 함수 추가
    var paramsObj = url.substring(url.indexOf('?') + 1).split('&');
    for (var i = 0, result = {}; i < paramsObj.length; i++) {
            paramsObj[i] = paramsObj[i].split('=');
            result[paramsObj[i][0]] = decodeURIComponent(paramsObj[i][1]);
    }
    // console.log(result);
    return result;
};
function shuffle(array){ //2022-07-07 GGCN45 : array => random Array
    var currentIdx = array.length,
        randomIdx;
    while (currentIdx != 0) {
        randomIdx = Math.floor(Math.random() * currentIdx);
        currentIdx --;
        [array[currentIdx], array[randomIdx]] = [array[randomIdx], array[currentIdx]];
    }

    return array;
};


$(document).on('change','input[name="finAllCheck"]',function(){
    if($('input[name="finAllCheck"]').is(':checked')){
        $('input[name="finchk"]').prop('checked',true);
    }else{
        $('input[name="finchk"]').prop('checked',false);
    }
});

$(document).on('change','input[name="finchk"]',function(){
    if($('input[name="finchk"]:checked').length == $('input[name="finchk"]').length){
        $('input[name="finAllCheck"]').prop('checked',true);
    }else{
        $('input[name="finAllCheck"]').prop('checked',false);
    }
});


$(document).on('change','.finAllCheck',function(){
    if($('.finAllCheck').is(':checked')){
        $('.finchk').prop('checked',true);
    }else{
        $('.finchk').prop('checked',false);
    }
});

$(document).on('change','.finchk',function(){
    if( $('.finchk:checked').length == $('.finchk').length){
        $('.finAllCheck').prop('checked',true);
        $('input[name="finAllCheck"]').prop('checked',true);
    }else{
        $('.finAllCheck').prop('checked',false);
    }
});

$(document).on('change', function(){
    if($ ('input[name="finchk"]:checked').length == $('input[name="finchk"]').length ) {
        $('input[name="finAllCheck"]').prop('checked',true);
    }else {
        $('input[name="finAllCheck"]').prop('checked',false);
    }
})