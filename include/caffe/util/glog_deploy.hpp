#ifndef _LOGGING_H_
#ifndef _GLOG_DEPLOY_HPP_
#define _GLOG_DEPLOY_HPP_

#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <sstream>

#if defined(__linux__) && defined(__GNUC__)
#include <execinfo.h>
#endif

class GlogDeoloyNullStream: public std::ostream {
public:
	GlogDeoloyNullStream(): std::ostream(nullptr) {
	}
	template <typename T>
	inline GlogDeoloyNullStream& operator<<(T const & x) {
		return *this;
	}
};
static GlogDeoloyNullStream glog_deploy_null_stream;

class GlogDeployLogMessage {
public:
	GlogDeployLogMessage(const std::string& level): level(level),
			stream(enable?std::cerr:glog_deploy_null_stream) {
	}
	~GlogDeployLogMessage() {
		if(level=="FATAL") {
#if defined(__linux__) && defined(__GNUC__)
			const int max_trace_num=20;
			void *buffer[max_trace_num];
			int trace_num=backtrace(buffer, max_trace_num);
			char **trace_string=backtrace_symbols(buffer, trace_num);
			std::cerr<<"Call stack trace:"<<std::endl;
			for(size_t i=0;i<trace_num;i++) {
				std::cerr<<std::string(trace_string[i])<<std::endl;
			}
			free(trace_string);
#else
			std::cerr<<"Call stack trace is not available."
#endif
			abort();
		}
		stream<<std::endl; 
	}
	static bool enable;
	std::ostream& stream;
	
private:
	std::string level;
};

#define LOG(type) GlogDeployLogMessage(#type).stream <<"[" << #type << "]\t[" << __FILE__ << "]\t[" << __LINE__ << "]\t"
#ifdef DEBUG
#define DLOG(type) LOG(type)
#else 
#define DLOG(type) if(true) {} else LOG(type)
#endif

#define CHECK(exp) if(exp) {} else LOG(FATAL) << "Check failed: " << #exp
#ifdef DEBUG
#define DCHECK(exp) CHECK(exp)
#else 
#define DCHECK(exp) if(true) {} else CHECK(exp)
#endif

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x) CHECK((x) != (NULL))

#define DCHECK_EQ(x, y) DCHECK((x) == (y))
#define DCHECK_LT(x, y) DCHECK((x) < (y))
#define DCHECK_GT(x, y) DCHECK((x) > (y))
#define DCHECK_LE(x, y) DCHECK((x) <= (y))
#define DCHECK_GE(x, y) DCHECK((x) >= (y))
#define DCHECK_NE(x, y) DCHECK((x) != (y))
#define DCHECK_NOTNULL(x) DCHECK((x) != (NULL))

#endif  // _GLOG_DEPLOY_HPP_
#endif  // _LOGGING_H_
