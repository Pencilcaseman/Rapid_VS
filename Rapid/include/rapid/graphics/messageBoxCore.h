#pragma once

#include "../internal.h"
#include <WinUser.h>

namespace rapid
{
	class RapidMessageBox
	{
	public:
		enum class MessageBoxType : int
		{
			ICON_ERROR = MB_ICONERROR,
			ICON_QUESTION = MB_ICONQUESTION,
			ICON_WARNING = MB_ICONWARNING,
			ICON_INFORMATION = MB_ICONINFORMATION,

			BUTTON_ABORD_RETRY_IGNORE = MB_ABORTRETRYIGNORE,
			BUTTON_CANCEL_TRY_CONTINUE = MB_CANCELTRYCONTINUE,
			BUTTON_HELP = MB_HELP,
			BUTTON_OK = MB_OK,
			BUTTON_OK_CANCEL = MB_OKCANCEL,
			BUTTON_RETRY_CANCEL = MB_RETRYCANCEL,
			BUTTON_YES_NO = MB_YESNO,
			BUTTON_YES_NO_CANCEL = MB_YESNOCANCEL,

			DEFAULT_FIRST = MB_DEFBUTTON1,
			DEFAULT_SECOND = MB_DEFBUTTON2,
			DEFAULT_THIRD = MB_DEFBUTTON3,

			RETURN_ABORT = IDABORT,
			RETURN_CANCEL = IDCANCEL,
			RETURN_CONTINUE = IDCONTINUE,
			RETURN_IGNORE = IDIGNORE,
			RETURN_NO = IDNO,
			RETURN_OK = IDOK,
			RETURN_RETRY = IDRETRY,
			RETURN_TRY_AGAIN = IDTRYAGAIN,
			RETURN_YES = IDYES
		};

		std::string title = "Rapid Message Box";
		std::string message = "Message Box";
		MessageBoxType icon = MessageBoxType::ICON_INFORMATION;
		MessageBoxType buttons = MessageBoxType::BUTTON_YES_NO_CANCEL;
		MessageBoxType defaultButton = MessageBoxType::DEFAULT_FIRST;

		RapidMessageBox() = default;

		RapidMessageBox(const std::string &t_,
						const std::string &message_ = "Message Box",
						const MessageBoxType icon_ = MessageBoxType::ICON_INFORMATION,
						const MessageBoxType buttons_ = MessageBoxType::BUTTON_YES_NO_CANCEL,
						const MessageBoxType defaultButton_ = MessageBoxType::DEFAULT_FIRST)
		{
			title = t_;
			message = message_;
			icon = icon_;
			buttons = buttons_;
			defaultButton = defaultButton_;
		}

		inline virtual bool pressAbort()
		{
			return true;
		}

		inline virtual bool pressCancel()
		{
			return true;
		}

		inline virtual bool pressContinue()
		{
			return true;
		}

		inline virtual bool pressIgnore()
		{
			return true;
		}

		inline virtual bool pressNo()
		{
			return true;
		}

		inline virtual bool pressOk()
		{
			return true;
		}

		inline virtual bool pressRetry()
		{
			return true;
		}

		inline virtual bool pressTryAgain()
		{
			return true;
		}

		inline virtual bool pressYes()
		{
			return true;
		}

		inline virtual bool error()
		{
			return true;
		}

		MessageBoxType display()
		{
			std::wstring wideMessage;
			wideMessage.assign(message.begin(), message.end());

			std::wstring wideTitle;
			wideTitle.assign(title.begin(), title.end());

		#ifdef __CUDA_ARCH__
			int msgBoxID = MessageBox(
				nullptr,
				(LPCSTR) wideMessage.c_str(),
				(LPCSTR) wideTitle.c_str(),
				(int) ((int) icon | (int) buttons | (int) defaultButton)
			);
		#else
			int msgBoxID = MessageBox(
				nullptr,
				wideMessage.c_str(),
				wideTitle.c_str(),
				(int) ((int) icon | (int) buttons | (int) defaultButton)
			);
		#endif

			bool errorOccured = false;

			switch (msgBoxID)
			{
				case (int) MessageBoxType::RETURN_ABORT:
					if (!pressAbort())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_CANCEL:
					if (!pressCancel())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_CONTINUE:
					if (!pressContinue())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_IGNORE:
					if (!pressIgnore())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_NO:
					if (!pressNo())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_OK:
					if (!pressOk())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_RETRY:
					if (!pressRetry())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_TRY_AGAIN:
					if (!pressTryAgain())
						errorOccured = true;
					break;
				case (int) MessageBoxType::RETURN_YES:
					if (!pressYes())
						errorOccured = true;
					break;
			}

			if (errorOccured)
			{
				rapidValidate(error(), "Message box failed");
			}

			return (MessageBoxType) msgBoxID;
		}
	};

	class RapidError : public RapidMessageBox
	{
	public:
		RapidError(const std::string &errorType_,
				   const std::string &errorMessage_)
		{
			title = errorType_;
			message = errorMessage_;

			buttons = RapidMessageBox::MessageBoxType::BUTTON_OK;
			icon = RapidMessageBox::MessageBoxType::ICON_ERROR;
		}

		bool pressOk() override
		{
			std::cerr << "Something went wrong\n";
			exit(1);

			return true;
		}
	};

	class RapidWarning : public RapidMessageBox
	{
	public:
		RapidWarning(const std::string &errorType_,
					 const std::string &errorMessage_,
					 const std::string &question = "Would you like to exit?")
		{
			title = errorType_;
			message = errorMessage_ + "\n\n" + question;

			buttons = RapidMessageBox::MessageBoxType::BUTTON_YES_NO;
			icon = RapidMessageBox::MessageBoxType::ICON_WARNING;
		}

		bool pressYes() override
		{
			std::cerr << "Warning failed\n";
			exit(1);

			return true;
		}

		bool pressNo() override
		{
			return true;
		}
	};
}
